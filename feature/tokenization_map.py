# @Time    : 2022/2/28 19:27
# @Author  : tk
# @FileName: tokenization_map.py

import unicodedata
import re

__all__ = ["rematch"]

def lowercase_and_normalize(text):
    """转小写，并进行简单的标准化
    """

    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text

def _cjk_punctuation():
    return u'\uff02\uff03\uff04\uff05\uff06\uff07\uff08\uff09\uff0a\uff0b\uff0c\uff0d\uff0f\uff1a\uff1b\uff1c\uff1d\uff1e\uff20\uff3b\uff3c\uff3d\uff3e\uff3f\uff40\uff5b\uff5c\uff5d\uff5e\uff5f\uff60\uff62\uff63\uff64\u3000\u3001\u3003\u3008\u3009\u300a\u300b\u300c\u300d\u300e\u300f\u3010\u3011\u3014\u3015\u3016\u3017\u3018\u3019\u301a\u301b\u301c\u301d\u301e\u301f\u3030\u303e\u303f\u2013\u2014\u2018\u2019\u201b\u201c\u201d\u201e\u201f\u2026\u2027\ufe4f\ufe51\ufe54\u00b7\uff01\uff1f\uff61\u3002'

def _is_punctuation(ch):
    """标点符号类字符判断（全/半角均在此内）
    提醒：unicodedata.category这个函数在py2和py3下的
    表现可能不一样，比如u'§'字符，在py2下的结果为'So'，
    在py3下的结果是'Po'。
    """
    code = ord(ch)
    return 33 <= code <= 47 or \
           58 <= code <= 64 or \
           91 <= code <= 96 or \
           123 <= code <= 126 or \
           unicodedata.category(ch).startswith('P')


def _stem(token):
    """获取token的“词干”（如果是##开头，则自动去掉##）
    """
    if token[:2] == '##':
        return token[2:]
    else:
        return token


def _is_space(ch):
    """空格类字符判断
    """
    return ch == ' ' or ch == '\n' or ch == '\r' or ch == '\t' or \
           unicodedata.category(ch) == 'Zs'


def _is_cjk_character(ch):
    """CJK类字符判断（包括中文字符也在此列）
    参考：https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    """
    code = ord(ch)
    return 0x4E00 <= code <= 0x9FFF or \
           0x3400 <= code <= 0x4DBF or \
           0x20000 <= code <= 0x2A6DF or \
           0x2A700 <= code <= 0x2B73F or \
           0x2B740 <= code <= 0x2B81F or \
           0x2B820 <= code <= 0x2CEAF or \
           0xF900 <= code <= 0xFAFF or \
           0x2F800 <= code <= 0x2FA1F


def _is_control(ch):
    """控制类字符判断
    """
    return unicodedata.category(ch) in ('Cc', 'Cf')



def _is_special(ch):
    """判断是不是有特殊含义的符号
    """
    return bool(ch) and (ch[0] == '[') and (ch[-1] == ']')


def _is_redundant(token):
    """判断该token是否冗余（默认情况下不可能分出来）
    """
    if len(token) > 1:
        for ch in _stem(token):
            if (_is_cjk_character(ch) or _is_punctuation(ch)):
                return True

def token_ids_decode(ids,token_dict_inv):
    """转为可读文本
    """

    tokens = [token_dict_inv[id] for id in ids]
    tokens = [token for token in tokens if not _is_special(token)]

    text, flag = '', False
    for i, token in enumerate(tokens):
        if token[:2] == '##':
            text += token[2:]
        elif len(token) == 1 and _is_cjk_character(token):
            text += token
        elif len(token) == 1 and _is_punctuation(token):
            text += token
            text += ' '
        elif i > 0 and _is_cjk_character(text[-1]):
            text += token
        else:
            text += ' '
            text += token

    text = re.sub(' +', ' ', text)
    text = re.sub('\' (re|m|s|t|ve|d|ll) ', '\'\\1 ', text)
    punctuation = _cjk_punctuation() + '+-/={(<['
    punctuation_regex = '|'.join([re.escape(p) for p in punctuation])
    punctuation_regex = '(%s) ' % punctuation_regex
    text = re.sub(punctuation_regex, '\\1', text)
    text = re.sub('(\d\.) (\d)', '\\1\\2', text)

    return text.strip()

def rematch(text, tokens,_do_lower_case):
    """给出原始的text和tokenize后的tokens的映射关系
    """
    if _do_lower_case:
        text = text.lower()

    normalized_text, char_mapping = '', []
    for i, ch in enumerate(text):
        if _do_lower_case:
            ch = lowercase_and_normalize(ch)
        ch = ''.join([
            c for c in ch
            if not (ord(c) == 0 or ord(c) == 0xfffd or _is_control(c))
        ])
        normalized_text += ch
        char_mapping.extend([i] * len(ch))

    text, token_mapping, offset = normalized_text, [], 0

    for i,token in enumerate(tokens):
        if token != '[UNK]' and _is_special(token):
            token_mapping.append([])
        else:
            token = _stem(token)
            try:
                start = text[offset:].index(token) + offset
                end = start + len(token)
            except Exception as e:
                start = offset
                end = start + 1

            token_mapping.append(char_mapping[start:end])
            offset = end

    return token_mapping