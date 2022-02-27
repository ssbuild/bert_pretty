import numpy as np
'''
    cls_softmax_decoding(batch_text, id2label, batch_logits,threshold=None)
    batch_text 文本list , 
    id2label 标签 list or dict
    batch_logits (batch,num_classes)
    threshold 阈值
'''

def cls_softmax_decoding(batch_text, id2label, batch_logits,threshold=None):
    formatted_outputs = []
    for (i, (text_raw, logits)) in enumerate(zip(batch_text, batch_logits)):
        if len(logits) > 1:
            if threshold is None:
                pred = np.argmax(logits,axis=-1)
                label = [id2label[pred]]
            else:
                pred = np.argsort(logits,axis=-1)[::-1]
                label =[id2label[l] for l in pred if logits[l] >= threshold]
        else:
            label = [id2label[logits]]
        formatted_outputs.append([label])
    return formatted_outputs

'''
    cls_sigmoid_decoding(batch_text, id2label, batch_logits,threshold=0.5)
    
    batch_text 文本list , 
    id2label 标签 list or dict
    batch_logits (batch,num_classes)
    threshold 阈值
'''


def cls_sigmoid_decoding(batch_text, id2label, batch_logits,threshold=0.5):
    assert threshold is not None
    formatted_outputs = []
    for (i, (text_raw, logits)) in enumerate(zip(batch_text, batch_logits)):
        pred = np.argsort(logits, axis=-1)[::-1]
        label = [id2label[l] for l in pred if logits[l] >= threshold]
        formatted_outputs.append(label)
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