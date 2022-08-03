# -*- coding: utf-8 -*-
# @Time    : 2022/8/3 14:13
# @Author  : tk
from functools import partial
import numpy as np
from .tokenization_map import rematch,token_ids_decode


__all__ = [
    'callback_char_level',
    'callback_word_level',
    'callback_char_level_input_ids_mask',
    'callback_word_level_input_ids_mask',
    'callback_char_level_input_ids_segment',
    'callback_word_level_input_ids_segment',
    'text_feature',
    'text_feature_char_level',
    'text_feature_word_level',
    'text_feature_char_level_input_ids_mask',
    'text_feature_word_level_input_ids_mask',
    'text_feature_char_level_input_ids_segment',
    'text_feature_word_level_input_ids_segment',
]


def get_word_token_ids(fn):
    def _get_word_token_ids(*args, **kargs):
        tokenizer, input_instance, max_seq_len, with_padding, with_token_mapping = args[:5]
        tokens = tokenizer.tokenize(input_instance)
        if len(tokens) > max_seq_len - 2:
            tokens = tokens[0:max_seq_len - 2]
        tokens.insert(0, "[CLS]")
        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        if with_token_mapping:
            mapping = rematch(input_instance, tokens, tokenizer.basic_tokenizer.do_lower_case)
        else:
            mapping = None
        return fn(tokenizer, input_ids, max_seq_len, with_padding, mapping)

    return _get_word_token_ids


def get_char_token_ids(fn):
    def _get_char_token_ids(*args, **kargs):
        tokenizer, input_instance, max_seq_len, with_padding = args[:4]
        word_list = list(input_instance)
        tokens = ["[CLS]"]
        for i, word in enumerate(word_list):
            token = tokenizer.tokenize(word)
            tokens.extend(token)
        if len(tokens) > max_seq_len - 1:
            tokens = tokens[0:max_seq_len - 1]
        tokens.append("[SEP]")
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        return fn(tokenizer, input_ids, max_seq_len, with_padding)

    return _get_char_token_ids


@get_char_token_ids
def callback_char_level(tokenizer, input_ids, max_seq_len, with_padding):
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    if with_padding:
        pad_val = tokenizer.vocab.get('[PAD]', 0)
        while len(input_ids) < max_seq_len:
            input_ids.append(pad_val)
            input_mask.append(0)
            segment_ids.append(0)
    return input_ids, input_mask, segment_ids


@get_char_token_ids
def callback_char_level_input_ids_mask(tokenizer, input_ids, max_seq_len, with_padding):
    input_mask = [1] * len(input_ids)
    if with_padding:
        pad_val = tokenizer.vocab.get('[PAD]', 0)
        while len(input_ids) < max_seq_len:
            input_ids.append(pad_val)
            input_mask.append(0)
    return input_ids, input_mask


@get_char_token_ids
def callback_char_level_input_ids_segment(tokenizer, input_ids, max_seq_len, with_padding):
    segment_ids = [0] * len(input_ids)
    if with_padding:
        pad_val = tokenizer.vocab.get('[PAD]', 0)
        while len(input_ids) < max_seq_len:
            input_ids.append(pad_val)
            segment_ids.append(0)
    return input_ids, segment_ids


@get_word_token_ids
def callback_word_level(tokenizer, input_ids, max_seq_len, with_padding, mapping):
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)
    if with_padding:
        pad_val = tokenizer.vocab.get('[PAD]', 0)
        while len(input_ids) < max_seq_len:
            input_ids.append(pad_val)
            input_mask.append(0)
            segment_ids.append(0)
    if mapping is not None:
        return input_ids, input_mask, segment_ids, mapping
    return input_ids, input_mask, segment_ids


@get_word_token_ids
def callback_word_level_input_ids_mask(tokenizer, input_ids, max_seq_len, with_padding, mapping):
    input_mask = [1] * len(input_ids)
    if with_padding:
        pad_val = tokenizer.vocab.get('[PAD]', 0)
        while len(input_ids) < max_seq_len:
            input_ids.append(pad_val)
            input_mask.append(0)
    if mapping is not None:
        return input_ids, input_mask, mapping
    return input_ids, input_mask


@get_word_token_ids
def callback_word_level_input_ids_segment(tokenizer, input_ids, max_seq_len, with_padding, mapping):
    segment_ids = [0] * len(input_ids)
    if with_padding:
        pad_val = tokenizer.vocab.get('[PAD]', 0)
        while len(input_ids) < max_seq_len:
            input_ids.append(pad_val)
            segment_ids.append(0)
    if mapping is not None:
        return input_ids, segment_ids, mapping
    return input_ids, segment_ids


'''
    text_feature

    tokenizer : FullTokenizer or your Tokenizer.
    max_seq_len: max length
    with_padding: whether padding to the max_seq_len , default padding to the batch max
    input_ids_callback: callback , example: tokens_word_mapper_callback ,tokens_char_mapper_callback
                    function has the param (tokenizer, text, max_seq_len,with_padding)
    input_ids_align_num:  对齐 回调返回的数目 tokens_char_mapper_callback , -1 all algin
'''


def text_feature(tokenizer, text_list: list,
                 max_seq_len: int = 128,
                 with_padding: bool = False,
                 with_token_mapping=False,
                 input_ids_callback=callback_word_level,
                 input_ids_align_num=-1, ):
    if text_list is None or not isinstance(text_list, list):
        return None
    all_ids = []
    r_max_len = 0
    is_muti_input = False

    for text in text_list:
        ids = input_ids_callback(tokenizer, text, max_seq_len, with_padding, with_token_mapping)
        if isinstance(ids, tuple):
            is_muti_input = True
            input_ids = ids[0]
            if not all_ids:
                for i in range(len(ids)):
                    all_ids.append([])
            for i in range(len(ids)):
                all_ids[i].append(ids[i])
        else:
            input_ids = ids
            if not all_ids:
                all_ids.append([])
            all_ids[0].append(input_ids)

        r_max_len = max(len(input_ids), r_max_len)

    if len(text_list) > 1:
        pad_val = tokenizer.vocab.get('[PAD]', 0)
        r_max_len = min(r_max_len, max_seq_len)
        item_num = len(all_ids) if is_muti_input else 1
        assert input_ids_align_num <= item_num
        n = input_ids_align_num if input_ids_align_num > 0 else item_num
        for i in range(n):
            all_ids[i] = list(map(lambda x: np.pad(x, (0, r_max_len - len(x)), constant_values=pad_val), all_ids[i]))
            all_ids[i] = np.asarray(all_ids[i], dtype=np.int32)
    return all_ids


text_feature_char_level = partial(text_feature, input_ids_callback=callback_char_level, input_ids_align_num=3)
text_feature_word_level = partial(text_feature, input_ids_callback=callback_word_level, input_ids_align_num=3)

text_feature_char_level_input_ids_mask = partial(text_feature, input_ids_callback=callback_char_level_input_ids_mask,
                                                 input_ids_align_num=2)
text_feature_word_level_input_ids_mask = partial(text_feature, input_ids_callback=callback_word_level_input_ids_mask,
                                                 input_ids_align_num=2)

text_feature_char_level_input_ids_segment = partial(text_feature,
                                                    input_ids_callback=callback_char_level_input_ids_segment,
                                                    input_ids_align_num=2)
text_feature_word_level_input_ids_segment = partial(text_feature,
                                                    input_ids_callback=callback_word_level_input_ids_segment,
                                                    input_ids_align_num=2)