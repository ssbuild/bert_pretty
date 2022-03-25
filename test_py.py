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
        text_feature_word_level_input_ids_segment


from bert_pretty.ner import load_label_bioes,load_label_bio,load_labels as ner_load_labels
from bert_pretty.ner import ner_crf_decoding,\
                            ner_pointer_decoding,\
                            ner_pointer_decoding_with_mapping,\
                            ner_pointer_double_decoding,ner_pointer_double_decoding_with_mapping

from bert_pretty.cls import cls_softmax_decoding,cls_sigmoid_decoding,load_labels as cls_load_labels


tokenizer = FullTokenizer(vocab_file=r'F:\pretrain\chinese_L-12_H-768_A-12\vocab.txt',do_lower_case=True)
text_list = ["你是谁123"]






#convert_to_ids 基础用法1
def test_feat1():
    feat = text_feature_char_level(tokenizer,text_list,max_seq_len=64,with_padding=True)
    print('char level',feat)
    feat = text_feature_char_level_input_ids_mask(tokenizer, text_list, max_seq_len=128, with_padding=False)
    print('char level', feat)
    feat = text_feature_char_level_input_ids_segment(tokenizer, text_list, max_seq_len=128, with_padding=False)
    print('char level', feat)

    #
    feat = text_feature_word_level(tokenizer,text_list,max_seq_len=128,with_padding=False,with_token_mapping=True)
    print('word level',feat)
    feat = text_feature_word_level_input_ids_mask(tokenizer, text_list, max_seq_len=128, with_padding=False,with_token_mapping=True)
    print('word level', feat)
    feat = text_feature_word_level_input_ids_segment(tokenizer, text_list, max_seq_len=128, with_padding=False,with_token_mapping=True)
    print('word level', feat)

    batch_input_ids,batch_seg_ids,batch_mapping = text_feature_word_level_input_ids_segment(tokenizer, text_list, max_seq_len=128, with_padding=False,
                                                     with_token_mapping=True)
    print('word level', batch_input_ids,batch_seg_ids,batch_mapping)

#convert_to_ids 基础用法2
def test_feat2():
    def my_input_callback1(tokenizer, input_instance, max_seq_len,with_padding,with_token_mapping):
        tokens = tokenizer.tokenize(input_instance)
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[0:max_seq_len - 2]
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)

        if with_padding:
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                segment_ids.append(0)
        return input_ids,segment_ids


    feat = text_feature(tokenizer,text_list,max_seq_len=128,with_padding=False,
                        input_ids_callback=my_input_callback1)
    print('自定义 callback1',feat)

    def my_input_callback2(tokenizer, input_instance, max_seq_len,with_padding,with_token_mapping):
        word_list = list(input_instance)
        tokens = ["[CLS]"]
        for i, word in enumerate(word_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        if len(tokens) > max_seq_len - 1:
            tokens = tokens[0:max_seq_len - 1]
        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if with_padding:
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
        return input_ids

    feat = text_feature(tokenizer, text_list, max_seq_len=128, with_padding=False,
                        input_ids_callback=my_input_callback2)
    print('自定义 callback2',feat)

    def my_input_callback3(tokenizer, input_instance, max_seq_len,with_padding,with_token_mapping):
        word_list = list(input_instance)
        tokens = ["[CLS]"]
        for i, word in enumerate(word_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        if len(tokens) > max_seq_len - 1:
            tokens = tokens[0:max_seq_len - 1]
        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        if with_padding:
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
        return input_ids,input_mask,segment_ids

    feat = text_feature(tokenizer, text_list, max_seq_len=128, with_padding=False,
                        input_ids_callback=my_input_callback3)
    print('自定义 callback3',feat)
    
    
    def my_input_callback4(tokenizer, input_instance, max_seq_len,with_padding,with_token_mapping):
        word_list = list(input_instance)
        tokens = ["[CLS]"]
        for i, word in enumerate(word_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        if len(tokens) > max_seq_len - 1:
            tokens = tokens[0:max_seq_len - 1]
        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = [0] * len(input_ids)
        if with_padding:
            while len(input_ids) < max_seq_len:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
        custom_data =['mapping']
        return input_ids,input_mask,segment_ids,custom_data

    feat = text_feature(tokenizer, text_list, max_seq_len=128, with_padding=False,
                        input_ids_callback=my_input_callback4,
                        input_ids_align_num=3,#只有前三个数据需要对齐 ，最后一个为自定义数据
                        )
    print('自定义 callback4',feat)


# labels = ['标签1','标签2']
# print(cls.load_labels(labels))
#
# print(ner.load_label_bio(labels))


'''
    # def ner_crf_decoding(batch_text, id2label, batch_logits, trans=None,batch_mapping=None):
    ner crf decode 解析crf序列  or 解析 已经解析过的crf序列

    batch_text input_instance list , 
    id2label 标签 list or dict
    batch_logits 为bert 预测结果 logits_all (batch,seq_len,num_tags) or (batch,seq_len)
    trans 是否启用trans预测 , 2D 
    batch_mapping 映射序列
'''

'''
    def ner_pointer_decoding(batch_text, id2label, batch_logits, threshold=1e-8,coordinates_minus=False)

    batch_text text list , 
    id2label 标签 list or dict
    batch_logits (batch,num_labels,seq_len,seq_len)
    threshold 阈值
    coordinates_minus
'''

'''
    def ner_pointer_decoding_with_mapping(batch_text, id2label, batch_logits, batch_mapping,threshold=1e-8,coordinates_minus=False)

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


    test_feat1()
    test_feat2()
    test_cls_decode()
    test_gpt_decode()





