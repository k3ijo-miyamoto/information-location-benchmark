"""Minimal hashing helpers used by publication training/replay code."""
from hashlib import sha256
import struct
import torch

def tensor_bytes(value):
    if value.dtype != torch.float64:
        raise ValueError("float64 digest required")
    numbers = value.detach().cpu().reshape(-1).tolist()
    return struct.pack("<" + "d" * len(numbers), *numbers)

def parameter_digest(model):
    digest = sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor_bytes(value))
    return digest.hexdigest()
