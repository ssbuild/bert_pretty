# -*- coding:utf-8 -*-
import setuptools
import sys
import platform
import os
import shutil

from setuptools import Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.build_py import build_py as _build_py

current_dir = os.path.dirname(os.path.abspath(__file__))


package_name = 'bert_pretty'
package_version = '0.1.0@post0'


title = 'bert_pretty is a text encoder and result decoder'

long_description_str = title
long_description_str += '\n\n'


def load_file(demo_file):
    with open(demo_file, mode='r', encoding='utf-8') as f:
        data_string = str(f.read())
    return data_string

data_string = load_file(os.path.join(current_dir,'test_py.py'))
long_description_str +=  '```py' + '\n' + data_string + '\n'+ '```' + '\n'


platforms_name = sys.platform + '_' + platform.machine()
class PrecompiledExtesion(Extension):
    def __init__(self, name):
        super().__init__(name, sources=[])

class build_ext(_build_ext):
    def build_extension(self, ext):
        if not isinstance(ext, PrecompiledExtesion):
            return super().build_extension(ext)




if __name__ == '__main__':
    setuptools.setup(
        platforms=platforms_name,
        name=package_name,
        version=package_version,
        author="ssbuild",
        author_email="9727464@qq.com",
        description=title,
        long_description_content_type='text/markdown',
        long_description=long_description_str,
        url="https://github.com/ssbuild/bert_pretty",
        package_dir={'bert_pretty': './bert_pretty'},
        packages=setuptools.find_packages('.'),   # 指定需要安装的模块
        python_requires='>=3, <4',
        #install_requires=['numpy>=1.18.0'],
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Developers',
            'Intended Audience :: Education',
            'Intended Audience :: Science/Research',
            'License :: OSI Approved :: Apache Software License',
            'Programming Language :: C++',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Programming Language :: Python :: 3.10',
            'Topic :: Scientific/Engineering',
            'Topic :: Scientific/Engineering :: Mathematics',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Software Development',
            'Topic :: Software Development :: Libraries',
            'Topic :: Software Development :: Libraries :: Python Modules',
        ],
        license='Apache 2.0',
        keywords=[package_name,'bert_pretty','bert text pretty','bert decording'],
    )