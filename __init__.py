# -*- coding: utf-8 -*-
from tokenization import FullTokenizer
from feature import callback_char_level, \
        callback_word_level,\
        callback_char_level_input_ids_mask,\
        callback_word_level_input_ids_mask, \
        callback_char_level_input_ids_segment, \
        callback_char_level_input_ids_segment, \
        text_feature, \
        text_feature_char_level,\
        text_feature_word_level,\
        text_feature_char_level_input_ids_mask, \
        text_feature_word_level_input_ids_mask, \
        text_feature_char_level_input_ids_segment, \
        text_feature_word_level_input_ids_segment


from ner import ner_crf_decoding,ner_pointer_decoding
from cls import cls_softmax_decoding,cls_sigmoid_decoding