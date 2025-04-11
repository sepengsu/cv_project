# model/__init__.py

import os
import importlib
import sys
import torch.nn as nn
from .utils import NotModel

# 하위 모듈 자동 import 및 클래스 등록
current_dir = os.path.dirname(__file__)
for filename in os.listdir(current_dir):
    if filename.endswith(".py") and filename != "__init__.py":
        module_name = f"model.{filename[:-3]}"
        module = importlib.import_module(module_name)

        # 해당 모듈 내 클래스들을 현재 네임스페이스에 등록
        for k in dir(module):
            obj = getattr(module, k)
            if (
                isinstance(obj, type)
                and issubclass(obj, nn.Module)
                and not issubclass(obj, NotModel)  # NotModel 상속한 클래스 제외
            ):
                globals()[k] = obj  # 현재 __init__ 모듈에 직접 등록

# 현재 모듈에서 모델 클래스만 추출
def get_model_classes():
    model_module = sys.modules[__name__]
    model_classes = {}
    for k in dir(model_module):
        obj = getattr(model_module, k)
        if (
            isinstance(obj, type)
            and issubclass(obj, nn.Module)
            and obj.__module__.startswith('model')
            and not issubclass(obj, NotModel)
        ):
            model_classes[k] = obj
    return model_classes

model_classes = get_model_classes()
MODEL_CLASS1 = dict(list(model_classes.items())[:len(model_classes) // 2])
MODEL_CLASS2 = dict(list(model_classes.items())[len(model_classes) // 2:])

__all__ = ["MODEL_CLASS1", "MODEL_CLASS2"]
