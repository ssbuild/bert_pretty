# -*- coding:utf-8 -*-
import numpy as np
from bert_text_pretty import cls,ner,relation,tokenization

'''
    简化文本特征化，解码
    https://github.com/ssbuild/bert_text_pretty.git
'''


text_list = ["你是谁123456","你是谁123456222222222222"]


tokenizer = tokenization.FullTokenizer(vocab_file=r'F:\pretrain\chinese_L-12_H-768_A-12\vocab.txt',do_lower_case=True)

feat = cls.cls_text_feature(tokenizer,text_list,max_seq_len=128,with_padding=False)
print(feat)

feat = ner.ner_text_feature(tokenizer,text_list,max_seq_len=128,with_padding=False)
print(feat)

feat = relation.re_text_feature(tokenizer,text_list,max_seq_len=128,with_padding=False)
print(feat)


labels = ['标签1','标签2']
print(cls.load_labels(labels))

print(ner.load_label_bio(labels))


# def ner_decoding(example_all, id2label, logits_all,trans=None) # crf 解码
'''
    example_all 文本list , 
    id2label 标签 list or dict
    logits_all 为bert 预测结果 (batch,seq_len,num_tags) or (batch,seq_len)
    trans 是否启用trans预测 , 2D 
    解析crf序列  or 解析 已经解析过的crf序列
     
'''


#ner_pointer_decoding(example_all, id2label, logits_all,threshold=0.) # 指针 解码
'''
   example_all 文本list , 
   id2label 标签 list or dict
   logits_all (batch,num_labels,seq_len,seq_len)
   threshold 阈值
  
'''



#  cls.cls_decoding(example_all,labels,logits) #分类
'''
   example_all 文本list , 
   id2label 标签 list or dict
   logits_all (batch,hidden)
   threshold 阈值

'''



# relation.re_decoding(example_all, id2spo, logits_all)  #关旭


