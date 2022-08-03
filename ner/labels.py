# @Time    : 2022/3/25 16:08
# @Author  : tk
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