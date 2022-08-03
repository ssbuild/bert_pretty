# @Time    : 2022/3/25 16:14
# @Author  : tk
# @FileName: ner_full_pointer.py

import numpy as np



'''
    def ner_pointer_decoding(batch_text, id2label, batch_logits, threshold=1e-8,coordinates_minus=False)

    batch_text text list , 
    id2label 标签 list or dict
    batch_logits (batch,num_labels,seq_len,seq_len)
    threshold 阈值
    coordinates_minus
'''


def ner_pointer_decoding(batch_text, id2label, batch_logits, threshold=1e-8,with_dict=True, coordinates_minus=False):
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
        if not with_dict:
            formatted_outputs.append(chunks)
            continue
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

def ner_pointer_decoding_with_mapping(batch_text, id2label, batch_logits, batch_mapping,threshold=1e-8,with_dict=True, coordinates_minus=False):
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

            if (start >= m_len -1 or start < 1) or (end >= m_len-1) or start > end:
                continue

            start = int(mapping[start][0])
            end = int(mapping[end][-1])
            if start > end or end >= t_length or start < 0:
                continue
            str_label = id2label[l]
            chunks.append((str_label, start, end, str(text_raw[start:end + 1])))
        if not with_dict:
            formatted_outputs.append(chunks)
            continue
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