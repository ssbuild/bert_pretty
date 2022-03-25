# @Time    : 2022/3/25 16:17
# @Author  : tk
# @FileName: ner_pointer_decode.py

import numpy as np



'''
    def ner_pointer_double_decoding(batch_text, id2label, batch_logits_start,batch_logits_end, coordinates_minus=False,threshold=0)

    batch_text text list , 
    id2label 标签 list or dict
    batch_logits_start (batch,seq_len,num_labels)
    batch_logits_end (batch,seq_len,num_labels)
    threshold 阈值
    coordinates_minus
'''


def ner_pointer_double_decoding(batch_text, id2label, batch_logits_start,batch_logits_end, coordinates_minus=False,threshold=0):
    formatted_outputs = []
    for (i, (text_raw, logits_start,logits_end)) in enumerate(zip(batch_text, batch_logits_start,batch_logits_end)):
        chunks = []
        ss = []
        es = []
        t_length = len(text_raw)

        for seq, l in zip(*np.where(logits_start > threshold)):
            seq = int(seq)
            l = int(l)
            if coordinates_minus:
                seq -= 1
            ss.append((seq, l))
        for seq, l in zip(*np.where(logits_end > threshold)):
            seq = int(seq)
            l = int(l)
            if coordinates_minus:
                seq -= 1
            es.append((seq, l))

        for s, l in ss:
            for e, l2 in es:
                if e < s:
                    continue
                if e >= t_length or s >= t_length:
                    continue
                if l != l2:
                    continue
                str_label = id2label[l]
                chunks.append((str_label, s, e,text_raw[s:e+1]))
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
    def ner_pointer_double_decoding_with_mapping(batch_text, id2label, batch_logits_start,batch_logits_end,batch_mapping, coordinates_minus=False,threshold=0)

    batch_text text list , 
    id2label 标签 list or dict
    batch_logits_start (batch,seq_len,num_labels)
    batch_logits_end (batch,seq_len,num_labels)
    batch_mapping
    threshold 阈值
    coordinates_minus
'''



def ner_pointer_double_decoding_with_mapping(batch_text, id2label, batch_logits_start,batch_logits_end,batch_mapping, coordinates_minus=False,threshold=0):
    formatted_outputs = []
    for (i, (text_raw, logits_start,logits_end)) in enumerate(zip(batch_text, batch_logits_start,batch_logits_end)):
        mapping = batch_mapping[i]
        chunks = []
        ss = []
        es = []
        t_length = len(text_raw)
        m_len = len(mapping)
        for seq, l in zip(*np.where(logits_start > threshold)):
            seq = int(seq)
            l = int(l)
            if coordinates_minus:
                seq -= 1
            if seq >= m_len:
                continue
            seq = int(mapping[seq][0])
            ss.append((seq, l))
        for seq, l in zip(*np.where(logits_end > threshold)):
            seq = int(seq)
            l = int(l)
            if coordinates_minus:
                seq -= 1
            if seq >= m_len:
                continue
            seq = int(mapping[seq][-1])
            es.append((seq, l))

        for s, l in ss:
            for e, l2 in es:
                if e < s:
                    continue
                if e >= t_length or s >= t_length:
                    continue
                if l != l2:
                    continue
                str_label = id2label[l]
                chunks.append((str_label, s, e,text_raw[s:e+1]))
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


