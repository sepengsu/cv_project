# model/__init__.py

import os
import importlib
import torch.nn as nn
from .utils import NotModel

# 현재 디렉토리의 .py 파일만 가져옴
modules = [f[:-3] for f in os.listdir(os.path.dirname(__file__)) 
           if f.endswith('.py') and f != '__init__.py']

for module in modules:
    imported = importlib.import_module(f'.{module}', package=__name__)
    globals().update({
        k: v for k, v in imported.__dict__.items()
        if isinstance(v, type)                          # class만 가져오기
        and issubclass(v, nn.Module)                    # nn.Module 기반
        and not issubclass(v, NotModel)                 # NotModel 상속받은 건 제외
    })
