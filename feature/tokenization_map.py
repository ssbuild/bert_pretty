# @Time    : 2022/2/28 19:27
# @Author  : tk
# @FileName: tokenization_map.py

import unicodedata

__all__ = ["rematch"]

def lowercase_and_normalize(text):
    """转小写，并进行简单的标准化
    """

    text = text.lower()
    text = unicodedata.normalize('NFD', text)
    text = ''.join([ch for ch in text if unicodedata.category(ch) != 'Mn'])
    return text


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
    for token in tokens:
        if _is_special(token):
            token_mapping.append([])
        else:
            token = _stem(token)
            start = text[offset:].index(token) + offset
            end = start + len(token)
            token_mapping.append(char_mapping[start:end])
            offset = end

    return token_mapping