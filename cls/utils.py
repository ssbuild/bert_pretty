import numpy as np
#粗粒度token
def cls_text_feature(
        tokenizer,
        text_list,
        max_seq_len=128,
        with_padding=False,
):
    b_input_ids,b_input_mask=[],[]
    r_max_len = 0
    for text in text_list:
        tokens = tokenizer.tokenize(text)
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[0:max_seq_len - 2]
        tokens.insert(0,"[CLS]")
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
    return b_input_ids,b_input_mask


def cls_decoding(example_all, id2label, logits_all):
    formatted_outputs = []
    for (i, (text_raw, logits)) in enumerate(zip(example_all, logits_all)):
        #label = get_tk_entities(text_raw,id2label,logits)
        if len(logits) > 1:
            logits = logits.argmax(axis=-1)
        label = id2label[logits]
        formatted_outputs.append([label])
    return formatted_outputs

def load_labels(label_file_or_list):
    if isinstance(label_file_or_list,list):
        labels = label_file_or_list
    else:
        with open(label_file_or_list, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            line = line.replace('\r\n', '')
            line = line.replace('\n', '')
            if line == '':
                continue
            labels.append(line)
    return labels