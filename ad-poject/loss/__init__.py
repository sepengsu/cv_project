# loss/__init__.py

import os
import importlib

# 현재 디렉토리의 .py 파일만 가져옴
modules = [f[:-3] for f in os.listdir(os.path.dirname(__file__)) 
           if f.endswith('.py') and f != '__init__.py']

for module in modules:
    imported = importlib.import_module(f'.{module}', package=__name__)
    globals().update({k: v for k, v in imported.__dict__.items() if not k.startswith('_')})
