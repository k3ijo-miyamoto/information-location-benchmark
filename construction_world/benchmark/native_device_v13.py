"""Explicit deterministic device selection for no-training v1.3 mechanics."""
import os
import platform
import torch
from .native_models_v13 import NativeModel


def configure(seed,device='cpu'):
    device = torch.device(device)
    if device.type not in ('cpu','cuda'): raise ValueError('unsupported device')
    if device.type=='cuda' and not torch.cuda.is_available(): raise ValueError('CUDA unavailable; no fallback')
    if device.type=='cuda' and (str(torch.__version__)!='2.13.0+cu130' or torch.version.cuda!='13.0' or platform.python_version()!='3.11.9'):
        raise ValueError('CUDA certification requires the frozen Python3.11.9/torch2.13.0+cu130/CUDA13.0 stack')
    value = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    if value != ':4096:8': raise ValueError('wrong cuBLAS workspace')
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.enabled = False
    torch.manual_seed(seed)
    return device


def build_model(family,seed,device='cpu'):
    device = configure(seed,device)
    # Identical seeded CPU initialization, then exact transfer to selected device.
    return NativeModel(family).to(device=device,dtype=torch.float64)
