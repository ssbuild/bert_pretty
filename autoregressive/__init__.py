# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 16:17
import typing
import numpy as np
# from ..tokenization import FullTokenizer
# from ..feature import token_ids_decode
# from bert_pretty import FullTokenizer

_DoneFlag_DONE = 0
_DoneFlag_CAN = 1
_DoneFlag_REDO = 2

# your model need 5 inputs [
# (('input_mask', 'input_type_ids', 'input_word_ids'),('input_mask', 'input_type_ids', 'input_word_ids'))
# 'temperature', 'top_k', 'top_p']
def callback_predict_fake(fn_args,
                          temperature,
                          top_k,
                          top_p,
                          *encoder_decoder_args):

    output = np.random.randint(670, 7000, size=(*encoder_decoder_args[0][0].shape, 21128))
    # results = model_pred.predict(x=inputs)
    return output

class GenerateStepInputBase:
    def __init__(self, tokenizer,
                 bos_token='[CLS]',
                 eos_token='[SEP]',
                 unk_token='[UNK]',
                 pad_token='[PAD]'
                 ):
        self.tokenizer = tokenizer
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.unk_token = unk_token
        self.pad_token = pad_token

        self.bos_token_id = self.tokenizer.vocab[bos_token]
        self.eos_token_id = self.tokenizer.vocab[eos_token]
        self.unk_token_id = self.tokenizer.vocab[unk_token]
        self.pad_token_id = self.tokenizer.vocab[pad_token]

        self.batch_input_ids_step = []
        self.batch_attention_mask_step = []
        self.batch_segment_ids_step = []

        self._batch_size = 0
        self.batch_input_ids = None
        self.batch_attention_mask =None
        self.batch_segment_ids = None

    def batch_step_zero(self):
        self.batch_input_ids_step.clear()
        self.batch_attention_mask_step.clear()
        self.batch_segment_ids_step.clear()
        for i in range(self.batch_size):
            self.batch_input_ids_step.append(0)
            self.batch_attention_mask_step.append(0)
            self.batch_segment_ids_step.append(0)

    def prepare(self, batch_text_or_tokens: typing.List):
        raise NotImplemented


    def prepare_step(self,idx,pred):
        raise NotImplemented

    def prepare_step_padding(self, idx):
        raise NotImplemented

    def get_inputs(self,idx_sels):
        return (self.batch_input_ids[idx_sels],self.batch_attention_mask[idx_sels],self.batch_segment_ids[idx_sels])

    @property
    def batch_size(self):
        return self._batch_size

    def __resort(self, x, y, z, pad_val):
        tmp = np.where(x != pad_val, -1, x)
        ids = np.argsort(tmp)
        x = np.take_along_axis(x, indices=ids, axis=-1)
        y = np.take_along_axis(y, indices=ids, axis=-1)
        z = np.take_along_axis(z, indices=ids, axis=-1)
        return x, y, z

    def forward_step(self,require_resort=False):
        batch_input_ids_step = np.expand_dims(np.asarray(self.batch_input_ids_step, dtype=np.int64),axis=1)
        batch_attention_mask_step = np.expand_dims(np.asarray(self.batch_attention_mask_step, dtype=np.int64),axis=1)
        batch_segment_ids_step = np.expand_dims(np.asarray(self.batch_segment_ids_step, dtype=np.int64),axis=1)

        self.batch_input_ids = np.concatenate((self.batch_input_ids, np.asarray(batch_input_ids_step, dtype=np.int32)), axis=1)
        self.batch_attention_mask = np.concatenate((self.batch_attention_mask, np.asarray(batch_attention_mask_step, dtype=np.int32)), axis=1)
        self.batch_segment_ids = np.concatenate((self.batch_segment_ids, np.asarray(batch_segment_ids_step, dtype=np.int32)), axis=1)
        if require_resort:
            self.batch_input_ids, self.batch_attention_mask, self.batch_segment_ids = self.__resort(
                self.batch_input_ids, self.batch_attention_mask, self.batch_segment_ids,self.pad_token_id)


class GenerateStepInputEncoderOnly(GenerateStepInputBase):
    def __init__(self,*args,**kwargs):
        super(GenerateStepInputEncoderOnly, self).__init__(*args, **kwargs)

    def prepare(self,batch_text_or_tokens: typing.List):
        b_tokens = []
        for text in batch_text_or_tokens:
            if isinstance(text, str):
                b_tokens.append(self.tokenizer.tokenize(text))
            else:
                b_tokens.append(list(text))

        self._batch_size = len(b_tokens)
        self.batch_input_ids = np.asarray([[self.bos_token_id] + self.tokenizer.convert_tokens_to_ids(input_tokens)
                                           for input_tokens in b_tokens],dtype=np.int64)
        self.batch_attention_mask = np.ones_like(self.batch_input_ids,dtype=np.int64)
        self.batch_segment_ids = np.zeros_like(self.batch_input_ids,dtype=np.int64)

    def prepare_step(self,idx,pred):
        self.batch_input_ids_step[idx] = pred
        self.batch_attention_mask_step[idx] = 1
        self.batch_segment_ids_step[idx] = 1

    def prepare_step_padding(self, idx):
        self.batch_input_ids_step[idx] = self.pad_token_id
        self.batch_attention_mask_step[idx] = 0
        self.batch_segment_ids_step[idx] = 0


class GenerateStepInputEncoder(GenerateStepInputBase):
    def __init__(self,*args,**kwargs):
        super(GenerateStepInputEncoder, self).__init__(*args,**kwargs)

    def prepare(self,batch_text_or_tokens: typing.List):
        b_tokens = []
        for text in batch_text_or_tokens:
            if isinstance(text, str):
                b_tokens.append(self.tokenizer.tokenize(text))
            else:
                b_tokens.append(list(text))
        self._batch_size = len(b_tokens)
        self.batch_input_ids = np.asarray([[self.bos_token_id] + self.tokenizer.convert_tokens_to_ids(input_tokens) + [self.eos_token_id]
                                           for input_tokens in b_tokens],dtype=np.int64)
        self.batch_attention_mask = np.ones_like(self.batch_input_ids,dtype=np.int64)
        self.batch_segment_ids = np.zeros_like(self.batch_input_ids,dtype=np.int64)

    def prepare_step(self,idx,pred):
        ...

    def prepare_step_padding(self, idx):
        ...

class GenerateStepInputDecoder(GenerateStepInputBase):
    def __init__(self,*args,**kwargs):
        super(GenerateStepInputDecoder, self).__init__(*args,**kwargs)

    def prepare(self,batch_text_or_tokens: typing.List):
        b_tokens = []
        for text in batch_text_or_tokens:
            if isinstance(text, str):
                b_tokens.append(self.tokenizer.tokenize(text))
            else:
                b_tokens.append(list(text))

        self._batch_size = len(b_tokens)
        self.batch_input_ids = np.asarray([[self.bos_token_id] for _ in range(self._batch_size)],dtype=np.int64)
        self.batch_attention_mask = np.ones_like(self.batch_input_ids,dtype=np.int64)
        self.batch_segment_ids = np.zeros_like(self.batch_input_ids,dtype=np.int64)

    def prepare_step(self,idx,pred):
        self.batch_input_ids_step[idx] = pred
        self.batch_attention_mask_step[idx] = 1
        self.batch_segment_ids_step[idx] = 1

    def prepare_step_padding(self, idx):
        self.batch_input_ids_step[idx] = self.pad_token_id
        self.batch_attention_mask_step[idx] = 0
        self.batch_segment_ids_step[idx] = 0



class GenerateStepInputWrapper:
    def __init__(self, tokenizer,with_decoder,
                 bos_token='[CLS]',
                 eos_token='[SEP]',
                 unk_token='[UNK]',
                 pad_token='[PAD]'
                 ):
        if with_decoder:
            self.encoder_input = GenerateStepInputEncoder(tokenizer,
                                                          bos_token=bos_token,
                                                          eos_token=eos_token,
                                                          unk_token=unk_token,
                                                          pad_token=pad_token,
                                                          )
            self.decoder_input = GenerateStepInputDecoder(tokenizer,
                                                          bos_token=bos_token,
                                                          eos_token=eos_token,
                                                          unk_token=unk_token,
                                                          pad_token=pad_token,
                                                          )
        else:
            self.encoder_input = GenerateStepInputEncoderOnly(tokenizer,
                                                              bos_token=bos_token,
                                                              eos_token=eos_token,
                                                              unk_token=unk_token,
                                                              pad_token=pad_token,
                                                              )
            self.decoder_input = None
    def prepare(self,batch_text_or_tokens: typing.List):
        self.encoder_input.prepare(batch_text_or_tokens)
        if self.decoder_input is not None:
            self.decoder_input.prepare(batch_text_or_tokens)

    def batch_step_zero(self):
        self.encoder_input.batch_step_zero()
        if self.decoder_input is not None:
            self.decoder_input.batch_step_zero()

    @property
    def batch_size(self):
        return self.encoder_input.batch_size

    def decoder_batch_input_ids(self):
        if self.decoder_input is not None:
            return self.decoder_input.batch_input_ids
        return self.encoder_input.batch_input_ids

    def get_inputs(self,idx_sels):
        if self.decoder_input is not None:
            return (self.encoder_input.get_inputs(idx_sels),self.decoder_input.get_inputs(idx_sels))
        return (self.encoder_input.get_inputs(idx_sels),)



    def prepare_step(self,idx,pred):
        self.encoder_input.prepare_step(idx,pred)
        if self.decoder_input is not None:
            self.decoder_input.prepare_step(idx,pred)

    def prepare_step_padding(self, idx):
        self.encoder_input.prepare_step_padding(idx)
        if self.decoder_input is not None:
            self.decoder_input.prepare_step_padding(idx)

    def forward_step(self, require_resort=False):
        self.encoder_input.forward_step(require_resort)
        if self.decoder_input is not None:
            self.decoder_input.forward_step(require_resort)

class GenerateMinMax:
    def __init__(self, tokenizer,with_decoder=False,
                 bos_token='[CLS]',
                 eos_token='[SEP]',
                 unk_token='[UNK]',
                 pad_token='[PAD]'
                 ):
        self.tokenizer = tokenizer
        self.input_step_instance = GenerateStepInputWrapper(tokenizer,with_decoder=with_decoder,
                                                            bos_token=bos_token,
                                                            eos_token=eos_token,
                                                            unk_token=unk_token,
                                                            pad_token=pad_token,
                                                            )


    def token_decode(self, logit):
        tokenizer = self.tokenizer
        token: str = tokenizer.inv_vocab[logit]
        if tokenizer.basic_tokenizer.do_lower_case:
            token = token.lower()
        if token.startswith('##'):
            token = token[2:]
        return token

    def decode(
            self,
            text: typing.AnyStr,
            fn_predict_call=callback_predict_fake,
            fn_args=None,
            max_length=64,
            temperature=1.0,
            top_k=3,
            top_p=1.0,
            ignore_tokens: typing.Union[typing.List,typing.Tuple]=('[UNK]',),
            end_tokens: typing.Union[typing.List,typing.Tuple]=('[SEP]',),
            num=10,
            max_retrey=3,
    ):
        assert num > 0
        batch_text_or_tokens = [text for _ in range(num)]
        return self.batch_decode(batch_text_or_tokens=batch_text_or_tokens,
                            fn_predict_call=fn_predict_call,
                            fn_args=fn_args,
                            max_length=max_length,
                            temperature=temperature,
                            top_k=top_k,
                            top_p=top_p,
                            ignore_tokens=ignore_tokens,
                            end_tokens=end_tokens,
                            max_retrey=max_retrey,
                            )

    def pre_process_logits(self, logits,temperature,top_k,top_p):
        # sample_func = lambda p: np.random.choice(len(p), p=p)  # 按概率采样函数
        # preds = np.apply_along_axis(sample_func, 1, logits)[0]
        # logits = softmax(logits[0],axis=-1)
        # print(np.where(logits >0))
        # preds = np.random.multinomial(4000, logits , size=1)
        # preds = np.argmax(logits[0],axis=-1)
        logits /= temperature
        logits = top_k_logits(logits, top_k)
        logits = top_p_logits(logits, top_p)
        return logits

    def post_process_logits(self,logits,ignore_tokens):
        for token in ignore_tokens:
            logits[self.tokenizer.vocab[token.upper()]] /= 10.0
        values_ids = np.where(logits > 1e-10)[0]
        values = softmax(logits[values_ids])
        values = np.clip(values, 0, 1)
        if len(values_ids):
            pred = np.random.choice(values_ids, p=values)
        else:
            pred = self.input_step_instance.encoder_input.eos_token_id
        return pred

    def batch_decode(
            self,
            batch_text_or_tokens: typing.List,
            fn_predict_call=callback_predict_fake,
            fn_args=None,
            max_length=64,
            temperature=1.0,
            top_k=3,
            top_p=1.0,
            ignore_tokens: typing.Union[typing.List,typing.Tuple]=('[UNK]',),
            end_tokens: typing.Union[typing.List,typing.Tuple] = ('[SEP]',),
            max_retrey=3,
    ):
        tokenizer = self.tokenizer
        if tokenizer.basic_tokenizer.do_lower_case:
            ignore_tokens = tuple(_.lower() for _ in ignore_tokens)
            end_tokens = tuple(_.lower() for _ in end_tokens)

        self.input_step_instance.prepare(batch_text_or_tokens)
        bs = self.input_step_instance.batch_size

        b_max_retrey = [max_retrey] * bs
        text_results = [[] for _ in range(bs)]

        starts = [len(self.input_step_instance.decoder_batch_input_ids()[_]) - 1 for _ in range(bs)]
        ends = [max_length-1 for _ in range(bs)]
        flags = []
        for i in range(bs):
            flags.append(_DoneFlag_CAN if starts[i] < ends[i] else _DoneFlag_DONE)

        flags_require_resort = [False] * bs
        while any(flags):
            idx_sels = np.asarray(flags, dtype=np.bool)
            batch_preds = fn_predict_call(fn_args,
                                          temperature,
                                          top_k,
                                          top_p,
                                          *self.input_step_instance.get_inputs(idx_sels))
            if batch_preds is None:
                break
            self.input_step_instance.batch_step_zero()
            for i,idx in enumerate(np.where(idx_sels>0)[0]):
                starts[idx] += 1
                # logits = batch_preds[i, :starts[idx]]
                logits = batch_preds[i:i+1,starts[idx] - 1]
                logits = self.pre_process_logits(logits,temperature,top_k,top_p)
                logits = logits[0]
                pred = self.post_process_logits(logits,ignore_tokens)
                token = self.token_decode(pred)

                if ignore_tokens is not None and token in ignore_tokens:
                    while (ignore_tokens is not None and token in ignore_tokens) and b_max_retrey[idx] >1:
                        b_max_retrey[idx] -= 1
                        pred = self.post_process_logits(logits, ignore_tokens)
                        token = self.token_decode(pred)

                is_ignore_char = False
                if ignore_tokens is not None and token in ignore_tokens:
                    b_max_retrey[idx] -= 1
                    if b_max_retrey[idx] < 0:
                        flags[idx] = _DoneFlag_DONE
                    else:
                        #重排序
                        flags_require_resort[i] = True
                        is_ignore_char = True
                        flags[idx] = _DoneFlag_REDO
                    starts[idx] -= 1

                if end_tokens is not None and token in end_tokens:
                    is_ignore_char = True
                    flags[idx] = _DoneFlag_DONE
                    starts[idx] -= 1


                if starts[idx] >= ends[idx]:
                    flags[idx] = _DoneFlag_DONE

                if flags[idx] == _DoneFlag_CAN:
                    self.input_step_instance.prepare_step(idx, pred)
                else:
                    self.input_step_instance.prepare_step_padding(idx)
                if not is_ignore_char:
                    text_results[idx].append(token)

            for idx in range(bs):
                if not idx_sels[idx]:
                    self.input_step_instance.prepare_step_padding(idx)
            require_resort = any(flags_require_resort)
            self.input_step_instance.forward_step(require_resort)

        return text_results


def top_k_logits(logits, k):
    if k == 0:
        return logits

    values = np.sort(logits)
    min_values = values[:, -k, np.newaxis]
    return np.where(
        logits < min_values,
        np.ones_like(logits, dtype=logits.dtype) * -1e10,
        logits,
    )


def softmax(z,axis=-1):
    t = np.exp(z)
    a = np.exp(z) / np.sum(t, axis=axis,keepdims = True)
    return a

def top_p_logits(logits, p):
    """Nucleus sampling"""
    batch, _ = logits.shape
    sorted_logits = np.sort(logits,axis=-1)
    sorted_logits = sorted_logits[:, ::-1]
    cumulative_probs = np.cumsum(softmax(sorted_logits, axis=1), axis=-1)

    indices = np.stack([
        np.arange(0, batch),
        # number of indices to include
        np.maximum(np.sum(cumulative_probs <= p, axis=-1) - 1, 0),
    ], axis=-1)
    min_values = np.asarray([sorted_logits[tuple(indice)] for indice in indices])
    return np.where(
        logits < min_values[:,np.newaxis],
        np.ones_like(logits) * -1e10,
        logits,
    )