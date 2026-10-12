"""Atomic epoch checkpoints with strict run/data/code binding and safe torch loading."""
import io
import os
import random
import tempfile
from hashlib import sha256
from pathlib import Path
import torch

MAGIC = b'G13-STAGE-V-CHECKPOINT-02\n'


class Paused(Exception):
    """Requested stop after a durable epoch boundary; not a numerical failure."""


def save(path, binding, model, optimizer, next_epoch, total_loss, runtime_seconds):
    if not 0 <= next_epoch <= 100:
        raise ValueError('invalid epoch boundary')
    state = dict(version=2, binding=binding, next_epoch=next_epoch,
                 model=model.state_dict(), optimizer=optimizer.state_dict(),
                 cpu_rng=torch.get_rng_state(),
                 cuda_rng=torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
                 python_rng=random.getstate(), total_loss=total_loss.detach().cpu(),
                 runtime_seconds=runtime_seconds)
    buffer = io.BytesIO()
    torch.save(state, buffer)
    payload = buffer.getvalue()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.checkpoint-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(MAGIC + sha256(payload).hexdigest().encode('ascii') + b'\n' + payload)
            f.flush(); os.fsync(f.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def load(path, binding, model, optimizer):
    data = Path(path).read_bytes()
    if not data.startswith(MAGIC): raise ValueError('checkpoint format mismatch')
    expected, payload = data[len(MAGIC):].split(b'\n', 1)
    if sha256(payload).hexdigest().encode('ascii') != expected:
        raise ValueError('checkpoint checksum mismatch')
    state = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
    if state['version'] != 2 or state['binding'] != binding:
        raise ValueError('checkpoint run/data/code binding mismatch')
    if type(state['next_epoch']) is not int or not 0 <= state['next_epoch'] <= 100:
        raise ValueError('checkpoint epoch mismatch')
    if any(not bool(torch.isfinite(v).all()) for v in state['model'].values()):
        raise ValueError('nonfinite checkpoint parameters')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['cpu_rng'])
    if state['cuda_rng']:
        torch.cuda.set_rng_state_all(state['cuda_rng'])
    random.setstate(state['python_rng'])
    return state
