# -*- coding:utf-8 -*-
'''
    bert input_instance encode and result decode
    https://github.com/ssbuild/bert_pretty.git
'''
import numpy as np
#FullTokenizer is official and you can use your tokenization .
from bert_pretty import FullTokenizer,\
        text_feature, \
        text_feature_char_level,\
        text_feature_word_level,\
        text_feature_char_level_input_ids_mask, \
        text_feature_word_level_input_ids_mask, \
        text_feature_char_level_input_ids_segment, \
        text_feature_word_level_input_ids_segment, \
        seqs_padding,rematch


from bert_pretty.ner import load_label_bioes,load_label_bio,load_labels as ner_load_labels
from bert_pretty.ner import ner_crf_decoding,\
                            ner_pointer_decoding,\
                            ner_pointer_decoding_with_mapping,\
                            ner_pointer_double_decoding,ner_pointer_double_decoding_with_mapping

from bert_pretty.cls import cls_softmax_decoding,cls_sigmoid_decoding,load_labels as cls_load_labels


tokenizer = FullTokenizer(vocab_file=r'F:\pretrain\chinese_L-12_H-768_A-12\vocab.txt',do_lower_case=True)
text_list = ["你是谁123aa\ta嘂a","嘂adasd"]



def test():
    maxlen = 512
    do_lower_case = tokenizer.basic_tokenizer.do_lower_case
    inputs = [['[CLS]'] + tokenizer.tokenize(text)[:maxlen - 2] + ['[SEP]'] for text in text_list]
    mapping = [rematch(text, tokens, do_lower_case) for text, tokens in zip(text_list, inputs)]
    inputs = [tokenizer.convert_tokens_to_ids(input) for input in inputs]
    input_mask = [[1] * len(input) for input in inputs]
    input_segment = [[0] * len(input) for input in inputs]
    input_ids = seqs_padding(inputs)
    input_mask = seqs_padding(input_mask)
    input_segment = seqs_padding(input_segment)

    input_ids = np.asarray(input_ids, dtype=np.int32)
    input_mask = np.asarray(input_mask, dtype=np.int32)
    input_segment = np.asarray(input_segment, dtype=np.int32)

    print('input_ids\n', input_ids)
    print('mapping\n',mapping)
    print('input_mask\n',input_mask)
    print('input_segment\n',input_segment)
    print('\n\n')



def test_charlevel():
    do_lower_case = tokenizer.basic_tokenizer.do_lower_case
    maxlen = 512
    if do_lower_case:
        inputs = [['[CLS]'] + tokenizer.tokenize(text.lower())[:maxlen - 2] + ['[SEP]'] for text in text_list]
    else:
        inputs = [['[CLS]'] + tokenizer.tokenize(text)[:maxlen - 2] + ['[SEP]'] for text in text_list]
    inputs = [tokenizer.convert_tokens_to_ids(input) for input in inputs]
    input_mask = [[1] * len(input) for input in inputs]
    input_segment = [[0] * len(input) for input in inputs]
    input_ids = seqs_padding(inputs)
    input_mask = seqs_padding(input_mask)
    input_segment = seqs_padding(input_segment)

    input_ids = np.asarray(input_ids, dtype=np.int32)
    input_mask = np.asarray(input_mask, dtype=np.int32)
    input_segment = np.asarray(input_segment, dtype=np.int32)

    print('input_ids\n', input_ids)
    print('input_mask\n',input_mask)
    print('input_segment\n',input_segment)
    print('\n\n')

# labels = ['标签1','标签2']
# print(cls.load_labels(labels))
#
# print(ner.load_label_bio(labels))


'''
    # def ner_crf_decoding(batch_text, id2label, batch_logits, trans=None,batch_mapping=None,with_dict=True):
    ner crf decode 解析crf序列  or 解析 已经解析过的crf序列

    batch_text input_instance list , 
    id2label 标签 list or dict
    batch_logits 为bert 预测结果 logits_all (batch,seq_len,num_tags) or (batch,seq_len)
    trans 是否启用trans预测 , 2D 
    batch_mapping 映射序列
'''

'''
    def ner_pointer_decoding(batch_text, id2label, batch_logits, threshold=1e-8,coordinates_minus=False,with_dict=True)

    batch_text text list , 
    id2label 标签 list or dict
    batch_logits (batch,num_labels,seq_len,seq_len)
    threshold 阈值
    coordinates_minus
'''

'''
    def ner_pointer_decoding_with_mapping(batch_text, id2label, batch_logits, batch_mapping,threshold=1e-8,coordinates_minus=False,with_dict=True)

    batch_text text list , 
    id2label 标签 list or dict
    batch_logits (batch,num_labels,seq_len,seq_len)
    threshold 阈值
    coordinates_minus
'''


'''
    cls_softmax_decoding(batch_text, id2label, batch_logits,threshold=None)
    batch_text 文本list , 
    id2label 标签 list or dict
    batch_logits (batch,num_classes)
    threshold 阈值
'''

'''
    cls_sigmoid_decoding(batch_text, id2label, batch_logits,threshold=0.5)

    batch_text 文本list , 
    id2label 标签 list or dict
    batch_logits (batch,num_classes)
    threshold 阈值
'''


def test_cls_decode():
    num_label =3
    np.random.seed(123)
    batch_logits = np.random.rand(2,num_label)
    result = cls_softmax_decoding(text_list,['标签1','标签2','标签3'],batch_logits,threshold=None)
    print(result)


    batch_logits = np.random.rand(2,num_label)
    print(batch_logits)
    result = cls_sigmoid_decoding(text_list,['标签1','标签2','标签3'],batch_logits,threshold=0.5)
    print(result)


def test_gpt_decode():
    from bert_pretty.gpt import autoregressive_decode_batch, autoregressive_decode_once

    result = autoregressive_decode_batch(tokenizer, end_symbol=['$','[SEP]'], max_length=10, start_text='你是谁123', try_count=3)

    for i,text in enumerate(result):
        print(i,len(text),''.join(text))

    print()
    result = autoregressive_decode_once(tokenizer,end_symbol=['$','[SEP]'],
                                        special_redo_symbol=['[PAD]','[UNK]'],
                                        max_length=10, start_text='你是谁123',
                                        try_count=3)

    for i, text in enumerate(result):
        print(i, len(text), ''.join(text))

if __name__ == '__main__':

    test()
    test_charlevel()

    test_cls_decode()
    test_gpt_decode()





