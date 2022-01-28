import json
import numpy as np
import os

#细粒度char token
def re_text_feature(
        tokenizer,
        text_list,
        max_seq_len=128,
        with_padding=False,
):
    b_input_ids, b_input_mask = [], []
    r_max_len = 0
    for text in text_list:
        word_list = list(text)
        tokens = ["[CLS]"]
        for i, word in enumerate(word_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        if len(tokens) > max_seq_len - 1:
            tokens = tokens[0:max_seq_len - 1]
        tokens.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(tokens)
        if with_padding:
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
            assert len(input_ids) == max_seq_len
            assert len(input_mask) == max_seq_len

        b_input_ids.append(input_ids)
        b_input_mask.append(input_mask)
        r_max_len = max(len(input_ids), r_max_len)


    if len(text_list) > 1:
        r_max_len = min(r_max_len, max_seq_len)
        iter1 = map(lambda x: np.pad(x, (0, r_max_len - len(x))), b_input_ids)
        b_input_ids = np.asarray([x for x in iter1], dtype=np.int32)

        iter2 = map(lambda x: np.pad(x, (0, r_max_len - len(x))), b_input_mask)
        b_input_mask = np.asarray([x for x in iter2], dtype=np.int32)
    return b_input_ids, b_input_mask


def find_entity(text_raw, id_, predictions):
    """
    retrieval entity mention under given predicate id for certain prediction.
    this is called by the "decoding" func.
    """
    entity_list = []
    for i in range(len(predictions)):
        if [id_] in predictions[i]:
            j = 0
            while i + j + 1 < len(predictions):
                if [1] in predictions[i + j + 1]:
                    j += 1
                else:
                    break
            s = i
            e = i + j
            entity = ''.join(text_raw[s : e + 1])
            entity_list.append((entity,(s,e)))
    return list(set(entity_list))

def re_decoding(example_all, id2spo, logits_all):
    """
    model output logits -> formatted spo (as in data set file)
    """
    formatted_outputs = []
    for (i, (text_raw, logits)) in enumerate(zip(example_all, logits_all)):
        seq_len = len(text_raw)
        logits = logits[1:seq_len + 1]  # slice between [CLS] and [SEP] to get valid logits
        # logits[logits >= 0.5] = 1
        # logits[logits < 0.5] = 0

        predictions = []
        for token in logits:
            predictions.append(np.argwhere(token == 1).tolist())

        # format predictions into example-style output
        formatted_instance = {}
        # flatten predictions then retrival all valid subject id
        flatten_predictions = []
        for layer_1 in predictions:
            for layer_2 in layer_1:
                flatten_predictions.append(layer_2[0])

        real_num_label = len(id2spo['predicate'])
        subject_id_list = []
        for cls_label in list(set(flatten_predictions)):
            if 1 < cls_label < real_num_label and (cls_label + real_num_label - 2) in flatten_predictions:
                subject_id_list.append(cls_label)
        subject_id_list = list(set(subject_id_list))
        # fetch all valid spo by subject id
        spo_list = []
        for id_ in subject_id_list:

            subjects = find_entity(text_raw, id_, predictions)
            objects = find_entity(text_raw, id_ + real_num_label - 2, predictions)

            predicate = id2spo['predicate'][id_].split('_',1)[0]
            for subject_ in subjects:
                for object_ in objects:
                    spo_list.append(
                        {
                            predicate : [
                                {
                                    "entity":subject_[0],
                                    'pos':[int(subject_[1][0]),int(subject_[1][1])],
                                    'label': id2spo['subject_type'][id_]
                                },
                                {
                                    "entity": object_[0],
                                    'pos':[int(object_[1][0]),int(object_[1][1])],
                                    'label': id2spo['object_type'][id_]
                                },
                            ]
                        }
                    )

        re_list = spo_list
        entities = {}

        def add_entity(entity, s, e, entity_label):
            if entity_label not in entities:
                entities[entity_label] = {}
            o = entities[entity_label]
            if entity not in o:
                o[entity] = []
            if [s, e] not in o[entity]:
                o[entity].append([s, e])

        for spo in re_list:
            for re_name in spo:
                if len(spo[re_name]) != 2:
                    continue
                for i in range(2):
                    entity = spo[re_name][i]['entity']
                    s = spo[re_name][i]['pos'][0]
                    e = spo[re_name][i]['pos'][1]
                    entity_label = spo[re_name][i]['label']
                    add_entity(entity, s, e, entity_label)
        #formatted_instance['text'] = text_raw
        formatted_instance['entities'] = entities
        formatted_instance['re_list'] = spo_list
        formatted_outputs.append(formatted_instance)
    return formatted_outputs

def load_labels(path):
    with open(path, mode='r', encoding='utf-8', newline='\n') as f:
        lines = f.readlines()

    relation_map = {}
    for line in lines:
        jd = json.loads(line)
        if not id:
            continue
        relation_map[jd['predicate']] = jd

    label2id = {
        'O': 0,
        'I': 1,
    }
    for p in relation_map:
        relation = relation_map[p]
        object_type_list = relation['object_type']

        for object_type in object_type_list:
            if len(object_type_list) > 1:
                label2id[p + '_' + object_type] = len(label2id)
            else:
                label2id[p] = len(label2id)

    id2spo = {
        "predicate": ["empty", "empty"],
        "subject_type": ["empty", "empty"],
        "object_type": ["empty", "empty"],
    }
    for p in relation_map:
        relation = relation_map[p]
        subject_type = relation['subject_type']
        object_type_list = relation['object_type']
        for object_type in object_type_list:
            id2spo["subject_type"].append(subject_type)
            id2spo["object_type"].append(object_type)
            if len(object_type_list) > 1:
                real_p = p + '_' + object_type
            else:
                real_p = p
            id2spo["predicate"].append(real_p)
    return id2spo