import re
import numpy as np
import copy
from collections import namedtuple

__chunk__ = namedtuple('Chunk', ['s', 'e','label','text'])

class __chunk__:
    def __init__(self,s=-1, e=-1, label='', text=''):
        self.s = s
        self.e = e
        self.label = label
        self.text=text


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
    chunk = __chunk__(s=-1, e=-1, label='', text='')
    def reset_chunk(chunk: __chunk__):
        chunk.s = -1
        chunk.e = -1
        chunk.label = ''
        chunk.text = ''

    for indx in range(length):
        tag = id2label[pred[indx]]
        if tag.startswith("S-"):
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            chunk.s = indx
            chunk.e = chunk.s
            chunk.label = tag.split('-')[1]
            chunk.text = text[chunk.s:chunk.e + 1]
            chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
        elif tag.startswith("B-"):
            if chunk.e != -1:
                chunk.text = text[chunk.s:chunk.e + 1]
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
            chunk.s = indx
            chunk.label = tag.split('-')[1]
        elif tag.startswith('I-') and chunk.s != -1:
            _type = tag.split('-')[1]
            if _type == chunk.label:
                chunk.e = indx
            if indx == length - 1:
                if chunk.e != -1:
                    chunk.text = text[chunk.s:chunk.e + 1]
                    chunks.append(copy.deepcopy(chunk))
                    reset_chunk(chunk)
        elif tag.startswith('E-') and chunk.s != -1:
            _type = tag.split('-')[1]
            if _type == chunk.label:
                chunk.e = indx
                chunk.text = text[chunk.s:chunk.e + 1]
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
        else:
            if chunk.e != -1:
                chunk.text = text[chunk.s:chunk.e + 1]
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)

    # [tuple(chunk) for chunk in chunks]
    # [('案发时间', 0, 9, '2014年3月29日')]
    labels = {}
    for chunk in chunks:
        l = chunk.label
        if l not in labels:
            labels[l] = {}
        o = labels[l]
        txt = chunk.text
        if txt not in o:
            o[txt] = [[chunk.s, chunk.e]]
        else:
            o[txt].append([chunk.s, chunk.e])
    return labels

# ['[CLS]', '我', '是', '中', '国', 'ASDASDASD1122', 'hello', 'world', '[SEP]']
# [[], [0], [1], [2], [3], [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [18, 19, 20, 21, 22], [24, 25, 26, 27, 28], []]

def get_entities_with_mapping(text,id2label,pred,mapping):
    assert len(text) + 2 == len(mapping)
    chunks = []
    chunk = __chunk__(s=-1,e=-1,label='',text='')

    def reset_chunk(chunk :__chunk__):
        chunk.s = -1
        chunk.e = -1
        chunk.label=''
        chunk.text=''

    length = len(pred)
    def SetS(chunk :__chunk__,indx):
        chunk.s = mapping[indx][0]

    def SetE(chunk :__chunk__,indx):
        chunk.s = mapping[indx][-1]

    for indx in range(length):
        tag = id2label[pred[indx]]
        if tag.startswith("S-"):
            if chunk.e != -1:
                chunks.append(copy.deepcopy(chunk))
            SetS(chunk,indx)
            SetE(chunk, indx)
            chunk.label = tag.split('-')[1]
            chunk.text = text[chunk.s:chunk.e + 1]
            chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
        elif tag.startswith("B-"):
            if chunk.e != -1:
                chunk.text = text[chunk.s:chunk.e + 1]
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
            SetS(chunk,indx)
            chunk.label = tag.split('-')[1]
        elif tag.startswith('I-') and chunk.s != -1:
            _type = tag.split('-')[1]
            if _type == chunk.label:
                SetE(chunk, indx)
            if indx == length - 1:
                if chunk.e != -1:
                    chunk.text = text[chunk.s:chunk.e + 1]
                    chunks.append(copy.deepcopy(chunk))
                    reset_chunk(chunk)
        elif tag.startswith('E-') and chunk.s != -1:
            _type = tag.split('-')[1]
            if _type == chunk.label:
                SetE(chunk, indx)
                chunk.text = text[chunk.s:chunk.e + 1]
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)
        else:
            if chunk.e != -1:
                chunk.text = text[chunk.s:chunk.e + 1]
                chunks.append(copy.deepcopy(chunk))
            reset_chunk(chunk)

    #[tuple(chunk) for chunk in chunks]
    #[('案发时间', 0, 9, '2014年3月29日')]
    labels = {}
    for chunk in chunks:
        l = chunk.label
        if l not in labels:
            labels[l] = {}
        o = labels[l]
        txt = chunk.text
        if txt not in o:
            o[txt] = [[chunk.s,chunk.e]]
        else:
            o[txt].append([chunk.s,chunk.e])
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

'''
    ner crf decode 解析crf序列  or 解析 已经解析过的crf序列
    
    batch_text text list , 
    id2label 标签 list or dict
    batch_logits 为bert 预测结果 logits_all (batch,seq_len,num_tags) or (batch,seq_len)
    trans 是否启用trans预测 , 2D 
    batch_mapping 映射序列
'''
def ner_crf_decoding(batch_text, id2label, batch_logits, trans=None,batch_mapping=None):
    formatted_outputs = []
    for (i, (text_raw, logits)) in enumerate(zip(batch_text, batch_logits)):
        if np.ndim(logits) == 2:
            if trans is None:
                logits = logits[1:-1].argmax(axis=-1)
            else:
                logits = viterbi_decode(logits,trans)[1:-1]
        else:
            logits = logits[1:-1]
        if batch_mapping:
            mapping = batch_mapping[i]
        else:
            mapping = None
        if mapping is None:
            label = get_entities(text_raw,id2label,logits)
        else:
            label = get_entities_with_mapping(text_raw,id2label,logits,mapping=mapping)
        formatted_outputs.append(label)
    return formatted_outputs


'''
    def ner_pointer_decoding(batch_text, id2label, batch_logits, threshold=1e-8,coordinates_minus=False)
    
    batch_text text list , 
    id2label 标签 list or dict
    batch_logits (batch,num_labels,seq_len,seq_len)
    threshold 阈值
    coordinates_minus
'''

def ner_pointer_decoding(batch_text, id2label, batch_logits, threshold=1e-8,coordinates_minus=False):
    batch_logits[:, :, [0, -1]] -= np.inf
    batch_logits[:, :, :, [0, -1]] -= np.inf
    formatted_outputs = []
    for (i, (text_raw, logits)) in enumerate(zip(batch_text, batch_logits)):
        chunks = []
        t_length = len(text_raw)
        for l, start, end in zip(*np.where(logits > threshold)):
            start = int(start)
            end = int(end)
            if coordinates_minus:
                start -= 1
                end -= 1
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


'''
    def ner_pointer_decoding_with_mapping(batch_text, id2label, batch_logits, batch_mapping,threshold=1e-8,coordinates_minus=False)

    batch_text text list , 
    id2label 标签 list or dict
    batch_logits (batch,num_labels,seq_len,seq_len)
    threshold 阈值
    coordinates_minus
'''

def ner_pointer_decoding_with_mapping(batch_text, id2label, batch_logits, batch_mapping,threshold=1e-8,coordinates_minus=False):
    batch_logits[:, :, [0, -1]] -= np.inf
    batch_logits[:, :, :, [0, -1]] -= np.inf
    formatted_outputs = []
    for (i, (text_raw, logits)) in enumerate(zip(batch_text, batch_logits)):
        mapping = batch_mapping[i]
        chunks = []
        t_length = len(text_raw)
        m_len = len(mapping)
        for l, start, end in zip(*np.where(logits > threshold)):
            start = int(start)
            end = int(end)
            if coordinates_minus:
                start -= 1
                end -= 1

            if (start >= m_len or start < 0) or (end > m_len) or start > end:
                continue

            start = int(mapping[start][0])
            end = int(mapping[end][-1])
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
