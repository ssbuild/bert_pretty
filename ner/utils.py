# @Time    : 2022/3/25 16:12
# @Author  : tk
# @FileName: utils.py

from collections import namedtuple

__chunk__ = namedtuple('Chunk', ['s', 'e','label','text'])

class __chunk__:
    def __init__(self,s=-1, e=-1, label='', text=''):
        self.s = s
        self.e = e
        self.label = label
        self.text=text