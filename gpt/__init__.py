# -*- coding: utf-8 -*-
# @Time    : 2022/3/1 16:17
# @Author  : wyw

import numpy as np
from bert_pretty.feature import token_ids_decode

from enum import Enum
class _DoneFlag_(Enum):
    DONE = 0
    CAN = 1
    REDO = 2

# your model need 5 inputs ['input_mask', 'input_type_ids', 'input_word_ids', 'temperature', 'top_k', 'top_p']
def callback_autoregressive_predict_fake(fn__args,b_input_ids, b_input_mask,b_seg_ids, b_temperature, b_top_k, b_top_p):
    results = np.random.randint(10000, 15000, size=(len(b_input_ids), 1))
    # results = model_pred.predict(x=inputs)
    batch_preds = results
    return batch_preds


def autoregressive_decode_batch(
        tokenizer,
        end_symbol : list or str,
        fn_predict_call=callback_autoregressive_predict_fake,
        fn_args=None,
        max_length=64,
        start_text='',
        try_count=10,
        temperature=1.0,
        top_k=3,
        top_p=1.0):
    if end_symbol is not None and not isinstance(end_symbol,list):
        end_symbol = [end_symbol]


    assert try_count > 0
    inital_tokens = ['[CLS]'] + tokenizer.tokenize(start_text)

    b_input_tokens= np.repeat([inital_tokens], try_count, axis=0)
    b_input_ids = [tokenizer.convert_tokens_to_ids(input_tokens) for input_tokens in b_input_tokens]
    b_input_mask = np.ones_like(b_input_ids)
    b_seg_ids = np.zeros_like(b_input_ids)
    b_temperature = [[temperature]] * len(b_input_ids)
    b_top_k = [[top_k]] * len(b_input_ids)
    b_top_p = [[top_p]] * len(b_input_ids)

    b_input_ids = np.asarray(b_input_ids, dtype=np.int32)
    b_input_mask = np.asarray(b_input_mask, dtype=np.int32)
    b_seg_ids = np.asarray(b_seg_ids, dtype=np.int32)
    b_temperature = np.asarray(b_temperature, dtype=np.float32)
    b_top_k = np.asarray(b_top_k, dtype=np.int32)
    b_top_p = np.asarray(b_top_p, dtype=np.float32)

    all_results = np.empty(try_count, dtype=object)
    text_results = [list(start_text) for _ in range(try_count)]
    starts = [len(inital_tokens) for _ in range(try_count)]
    ends = [max_length for _ in range(try_count)]
    flags = [_DoneFlag_.CAN for _ in range(try_count)]
    current_pred_token = ['' for _ in range(try_count)]
    get_out_seqs = [i for i in range(try_count)]

    while len(flags) and any(flags):
        batch_preds = fn_predict_call(fn_args,b_input_ids, b_input_mask,b_seg_ids, b_temperature, b_top_k, b_top_p)
        if batch_preds is None:
            break
        for idx in range(len(b_input_ids)):
            starts[idx] += 1
            preds = batch_preds[idx]
            tokens = token_ids_decode(preds,tokenizer.inv_vocab)
            if not tokens:
                tokens = ['[UNK]']
            current_pred_token[idx] = tokens[0]
            if end_symbol is not None and tokens[0] in end_symbol:
                flags[idx] = _DoneFlag_.DONE
            if starts[idx] >= ends[idx]:
                flags[idx] = _DoneFlag_.DONE

        b_input_ids = np.concatenate((b_input_ids, batch_preds), axis=1)
        b_input_mask = np.ones_like(b_input_ids)
        b_seg_ids = np.concatenate((b_seg_ids, np.ones(shape=(len(batch_preds), 1), dtype=np.int32)), axis=1)

        for idx in range(len(batch_preds)):
            if flags[idx] == _DoneFlag_.DONE:
                real_idx = get_out_seqs[idx]
                all_results[real_idx] = text_results[idx]
            elif flags[idx] == _DoneFlag_.CAN:
                text_results[idx].append(current_pred_token[idx])


        idx_remove = [idx for idx in range(len(b_input_ids)) if flags[idx] == _DoneFlag_.DONE]
        text_results = np.delete(text_results, idx_remove, axis=0).tolist()
        b_input_ids = np.delete(b_input_ids, idx_remove, axis=0)
        b_input_mask = np.delete(b_input_mask, idx_remove, axis=0)
        b_seg_ids = np.delete(b_seg_ids, idx_remove, axis=0)
        b_temperature = np.delete(b_temperature, idx_remove, axis=0)
        b_top_k = np.delete(b_top_k, idx_remove, axis=0)
        b_top_p = np.delete(b_top_p, idx_remove, axis=0)
        starts = np.delete(starts, idx_remove, axis=0)
        ends = np.delete(ends, idx_remove, axis=0)
        flags = np.delete(flags, idx_remove, axis=0)
        get_out_seqs = np.delete(get_out_seqs, idx_remove, axis=0)


    return all_results

# end_symbol=['$','[SEP]']
def autoregressive_decode_once(
        tokenizer,
        end_symbol : list or str,
        special_redo_symbol: list or str,
        fn_predict_call=callback_autoregressive_predict_fake,
        fn_args=None,
        max_length=64,
        start_text='',
        try_count=10,
        temperature=1.0,
        top_k=3,
        top_p=1.0,
):
    assert try_count > 0

    if end_symbol is not None and not isinstance(end_symbol,list):
        end_symbol = [end_symbol]

    if special_redo_symbol is not None and not isinstance(special_redo_symbol,list):
        special_redo_symbol = [special_redo_symbol]

    all_results = []

    inital_tokens = ['[CLS]'] + tokenizer.tokenize(start_text)
    raw_text_list = np.repeat([inital_tokens], 1, axis=0)


    for i in range(try_count):
        text_results = [list(start_text) for _ in range(1)]
        starts = [len(inital_tokens) for _ in range(1)]
        ends = [max_length for _ in range(1)]
        flags = [_DoneFlag_.CAN for _ in range(1)]
        current_pred_token = ['' for _ in range(1)]


        b_input_ids = [tokenizer.convert_tokens_to_ids(tokens) for tokens in raw_text_list]
        b_input_mask = np.ones_like(b_input_ids)
        b_seg_ids = np.zeros_like(b_input_ids)
        b_temperature = [[temperature]] * len(b_input_ids)
        b_top_k = [[top_k]] * len(b_input_ids)
        b_top_p = [[top_p]] * len(b_input_ids)

        b_input_ids = np.asarray(b_input_ids, dtype=np.int32)
        b_input_mask = np.asarray(b_input_mask, dtype=np.int32)
        b_seg_ids = np.asarray(b_seg_ids, dtype=np.int32)
        b_temperature = np.asarray(b_temperature, dtype=np.float32)
        b_top_k = np.asarray(b_top_k, dtype=np.int32)
        b_top_p = np.asarray(b_top_p, dtype=np.float32)


        while len(flags) and any(flags):
            batch_preds = fn_predict_call(fn_args,b_input_ids, b_input_mask,b_seg_ids, b_temperature, b_top_k, b_top_p)
            if batch_preds is None:
                break
            for idx in range(len(batch_preds)):
                starts[idx] += 1
                preds = batch_preds[idx]
                tokens = token_ids_decode(preds, tokenizer.inv_vocab)
                if not tokens:
                    tokens = ['[UNK]']
                if end_symbol is not None and tokens[0] in end_symbol:
                    flags[idx] = _DoneFlag_.DONE
                elif special_redo_symbol is not None and tokens[0] in special_redo_symbol:
                    starts[idx] -= 1
                    flags[idx] = _DoneFlag_.REDO

                current_pred_token[idx] = tokens[0]

                if starts[idx] >= ends[idx]:
                    flags[idx] = _DoneFlag_.DONE

            if flags[0] == _DoneFlag_.CAN:
                b_input_ids = np.concatenate((b_input_ids, batch_preds), axis=1)
                b_input_mask = np.ones_like(b_input_ids)
                b_seg_ids = np.concatenate((b_seg_ids, np.ones(shape=(len(batch_preds), 1), dtype=np.int32)), axis=1)

            for idx in range(len(batch_preds)):
                if flags[idx] == _DoneFlag_.DONE:
                    all_results.append(text_results[idx])
                elif flags[idx] == _DoneFlag_.REDO:
                    pass
                elif flags[idx] == _DoneFlag_.CAN:
                    text_results[idx].append(current_pred_token[idx])



            idx_remove = [idx for idx in range(len(b_input_ids)) if flags[idx] == _DoneFlag_.DONE]
            text_results = np.delete(text_results, idx_remove, axis=0).tolist()
            b_input_ids = np.delete(b_input_ids, idx_remove, axis=0)
            b_input_mask = np.delete(b_input_mask, idx_remove, axis=0)
            b_seg_ids = np.delete(b_seg_ids, idx_remove, axis=0)
            b_temperature = np.delete(b_temperature, idx_remove, axis=0)
            b_top_k = np.delete(b_top_k, idx_remove, axis=0)
            b_top_p = np.delete(b_top_p, idx_remove, axis=0)
            starts = np.delete(starts, idx_remove, axis=0)
            ends = np.delete(ends, idx_remove, axis=0)
            flags = np.delete(flags, idx_remove, axis=0)

    return all_results