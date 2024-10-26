import json
import subprocess
import torch
import os
import sys
from torch.backends import cudnn

if torch.cuda.is_available():
    print('CUDA 可用')
else:
    print("CUDA不可用，请确保是英伟达显卡并安装了CUDA11.8+版本")
    sys.exit()
    
if cudnn.is_available() and cudnn.is_acceptable(torch.tensor(1.).cuda()):
    print('cuDNN可用')
else:
    print('cuDNN不可用，请安装cuDNN')
    sys.exit()
