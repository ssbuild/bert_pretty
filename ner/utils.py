import re
import numpy as np

#细粒度char token
def ner_text_feature(
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


def process_result(text,id2label,prob):
    result = []
    for v in prob[1:len(text) + 1]:
        # print(int(v))
        # print(self.id2label[int(v)])
        result.append(id2label[int(v)])
    labels = {}
    start = None
    index = 0
    for w, t in zip("".join(text), result):
        if re.search("^[BS]", t):
            if start is not None:
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = text[start:index]
                    # print(te_, labels)
                    labels[label] = {te_: [[start, index - 1]]}
            start = index
            # print(start)
        if re.search("^O", t):
            if start is not None:
                # print(start)
                label = result[index - 1][2:]
                if labels.get(label):
                    te_ = str(text[start:index])
                    # print(te_, labels)
                    labels[label][te_] = [[start, index - 1]]
                else:
                    te_ = str(text[start:index])
                    # print(te_, labels)
                    labels[label] = {te_: [[start, index - 1]]}
            # else:
            #     print(start, labels)
            start = None
        index += 1
    if start is not None:
        # print(start)
        label = result[start][2:]
        if labels.get(label):
            te_ = str(text[start:index])
            # print(te_, labels)
            labels[label][te_] = [[start, index - 1]]
        else:
            te_ = str(text[start:index])
            # print(te_, labels)
            labels[label] = {te_: [[start, index - 1]]}
    # print(labels)
    return labels



def get_entities(text,id2label,pred):
    length = min(len(text), len(pred))
    chunks = []
    chunk = [-1, -1, -1,'']
    for indx in range(length):
        tag = id2label[pred[indx]]
        if tag.startswith("S-"):
            if chunk[2] != -1:
                chunks.append(chunk)
            chunk = [-1, -1, -1,'']
            chunk[1] = indx
            chunk[2] = indx
            chunk[0] = tag.split('-')[1]
            chunk[3] = text[chunk[1]:chunk[2] + 1]
            chunks.append(chunk)
            chunk = [-1, -1, -1,'']
        elif tag.startswith("B-"):
            if chunk[2] != -1:
                chunk[3] = text[chunk[1]:chunk[2] + 1]
                chunks.append(chunk)
            chunk = [-1, -1, -1,'']
            chunk[1] = indx
            chunk[0] = tag.split('-')[1]
        elif tag.startswith('I-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
            if indx == length - 1:
                if chunk[2] != -1:
                    chunk[3] = text[chunk[1]:chunk[2] + 1]
                    chunks.append(chunk)
        elif tag.startswith('E-') and chunk[1] != -1:
            _type = tag.split('-')[1]
            if _type == chunk[0]:
                chunk[2] = indx
                chunk[3] = text[chunk[1]:chunk[2] + 1]
                chunks.append(chunk)
            chunk = [-1, -1, -1, '']
        else:
            if chunk[2] != -1:
                chunk[3] = text[chunk[1]:chunk[2] + 1]
                chunks.append(chunk)
            chunk = [-1, -1, -1,'']

    #[tuple(chunk) for chunk in chunks]
    #[('案发时间', 0, 9, '2014年3月29日')]
    labels = {}
    for chunk in chunks:
        l = chunk[0]
        if l not in labels:
            labels[l] = {}
        o = labels[l]
        txt = chunk[3]
        if txt not in o:
            o[txt] = [[chunk[1],chunk[2]]]
        else:
            o[txt].append([chunk[1],chunk[2]])
    return labels


def viterbi_decode(score, transition_params):
    """Decode the highest scoring sequence of tags outside of TensorFlow.

    This should only be used at test time.

    Args:
      score: A [seq_len, num_tags] matrix of unary potentials.
      transition_params: A [num_tags, num_tags] matrix of binary potentials.

    Returns:
      viterbi: A [seq_len] list of integers containing the highest scoring tag
          indices.
      viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]

    for t in range(1, score.shape[0]):
        v = np.expand_dims(trellis[t - 1], 1) + transition_params
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    viterbi.reverse()
    return viterbi

#解析crf序列  or 解析 已经解析过的crf序列
#logits_all (batch,seq_len,num_tags) or (batch,seq_len)
def ner_decoding(example_all, id2label, logits_all,trans=None):
    formatted_outputs = []
    for (i, (text_raw, logits)) in enumerate(zip(example_all, logits_all)):
        if np.ndim(logits) == 2:
            if trans is None:
                logits = logits[1:-1].argmax(axis=-1)
            else:
                logits = viterbi_decode(logits,trans)[1:-1]
        else:
            logits = logits[1:-1]
        label = get_entities(text_raw,id2label,logits)
        formatted_outputs.append(label)
    return formatted_outputs

# 解析ner指针 (batch,num_labels,seq_len,seq_len)
def ner_pointer_decoding(example_all, id2label, logits_all, threshold=1e-8):
    formatted_outputs = []
    for (i, (text_raw, logits)) in enumerate(zip(example_all, logits_all)):
        chunks = []
        t_length = len(text_raw)
        for l, start, end in zip(*np.where(logits > threshold)):
            start -= 1
            end -= 1
            start = int(start)
            end = int(end)
            if start > end or end >= t_length or start < 0:
                continue
            str_label = id2label[l]
            chunks.append((str_label, start, end, str(text_raw[start:end + 1])))

        labels = {}
        for chunk in chunks:
            l = chunk[0]
            if l not in labels:
                labels[l] = {}
            o = labels[l]
            txt = chunk[3]
            if txt not in o:
                o[txt] = [[chunk[1], chunk[2]]]
            else:
                o[txt].append([chunk[1], chunk[2]])
        formatted_outputs.append(labels)
    return formatted_outputs

def load_label_bio(label_file_or_list):
    if isinstance(label_file_or_list,list):
        labels = label_file_or_list
    else:
        with open(label_file_or_list, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            line = line.replace('\r\n','')
            line = line.replace('\n', '')
            if line == '':
                continue
            labels.append(line)
    labels =['O'] + [i + '-' + j  for i in ['B','I'] for j in labels]
    #label2id ={j:i for i,j in enumerate(labels)}
    id2label={i:j for i,j in enumerate(labels)}
    return id2label

def load_label_bioes(label_file_or_list):
    if isinstance(label_file_or_list, list):
        labels = label_file_or_list
    else:
        with open(label_file_or_list, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            line = line.replace('\r\n','')
            line = line.replace('\n', '')
            if line == '':
                continue
            labels.append(line)
    labels =['O'] + [i + '-' + j  for i in ['B','I','E','S'] for j in labels]
    #label2id ={j:i for i,j in enumerate(labels)}
    id2label={i:j for i,j in enumerate(labels)}
    return id2label

def load_labels(label_file_or_list):
    if isinstance(label_file_or_list, list):
        labels = label_file_or_list
    else:
        with open(label_file_or_list, mode='r', encoding='utf-8') as f:
            lines = f.readlines()
        labels = []
        for line in lines:
            line = line.replace('\r\n','')
            line = line.replace('\n', '')
            if line == '':
                continue
            labels.append(line)
    id2label={i:j for i,j in enumerate(labels)}
    return id2label
