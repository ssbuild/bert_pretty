# -*- coding: utf-8 -*-
# @Time    : 2023/2/1 10:29

# -*- coding: utf-8 -*-
import numpy as np
from nn_sdk import *
config = {
    "model_dir": r'D:\onnx\unilm\best.onnx',
    "aes":{
        "use":False,
        "key":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
        "iv":bytes([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]),
    },
    "log_level": 8,# 0 fatal , 2 error , 4 warn, 8 info , 16 debug
    'engine':1, # 0 tensorflow,  1 onnx , 2  tensorrt , 3 fasttext
    "device_id": 0,
    'tf':{
        "ConfigProto": {
            "log_device_placement": False,
            "allow_soft_placement": True,
            "gpu_options": {"allow_growth": True},
            "graph_options":{
                "optimizer_options":{"global_jit_level": 1}
            },
        },
        "engine_major": 1, # tensorflow engine majar version
        "is_reset_graph": 1, # 1 reset_default_graph , 0 do nothing
        "model_type": 0,# 0 pb , 1 ckpt
        #配置pb模型
        "saved_model":{
            # model_type为 1 pb , 模型有效,
            # 模型是否是是否启用saved_model冻结 , 如果是,则 use=True并且配置tags
            # 普通 freeze pb , use = False
            'enable': False, # 是否启用saved_model
            'tags': ['serve'],
            'signature_key': 'serving_default',
        },
        "fastertransformer":{"enable": False}
    },
    'onnx':{
        'tensorrt': True, #是否启用tensorrt算子
    },
    'trt':{
        #pip install trt-sdk , support tensorrt 7.2 8.0 8.2 8.4 or more new
        "engine_major": 8,# 7 or 8
        "engine_minor": 0,
        "enable_graph": 0,
    },
    "graph": [
        {
            # 对于Bert模型 shape [max_batch_size,max_seq_lenth],
            # 其中max_batch_size 用于c++ java开辟输入输出缓存,输入不得超过max_batch_size，对于python没有作用，取决于上层用户真实输入
            # python 限制max_batch_size 在上层用户输入做 , dtype and shape are not necessary for python
            # 对于fasttext node 对应name可以任意写，但不能少
            # dtype must be in [int int32 int64 long longlong uint uint32 uint64 ulong ulonglong float float32 float64 double str]
              "input": [
                {"node":"input_ids","dtype":"int64",},
                {"node":"token_type_ids","dtype":"int64",},
            ],
            "output": [
                {
                    "node":"pred_ids","dtype":"int64",
                },
            ],
        }
    ]}



from autoregressive import GenerateMinMax
from tokenization import FullTokenizer

tokenizer = FullTokenizer(vocab_file=r'D:\onnx\unilm\vocab.txt',do_lower_case=True,)


seq_length = 256
sdk_inf = csdk_object(config)
if sdk_inf.valid():
    def fn_predict_call(fn_args,
                        b_temperature: float,
                        b_top_k: int,
                        b_top_p: int,
                        *encoder_decoder_args):
        net_stage = 0
        ret, outputs = sdk_inf.process(net_stage, encoder_decoder_args[0][0].astype(np.int64), encoder_decoder_args[0][2].astype(np.int64))
        logits = outputs[0]
        return logits


    generate = GenerateMinMax(tokenizer,
                              bos_token='[CLS]',
                              eos_token='[SEP]',
                              unk_token='[UNK]',
                              pad_token='[PAD]'
                              )

    result = generate.decode(text='[unused3]' + '<t>',
                    end_tokens=['$','[SEP]'],
                    max_length=100,
                    num=10,
                    fn_predict_call=fn_predict_call)

    for i, text in enumerate(result):
        print(i, len(text), ''.join(text))
    sdk_inf.close()