"""Frozen final-epoch Stage V training; no tuning, early stopping or Stage C entry."""
import time
from dataclasses import replace
import torch
from .native_device_v13 import build_model
from .native_no_training_gate_v13 import parameter_digest
from .native_seeds_v13 import derive
from .native_stage_v_batch_v13 import forward, select, stack
from .native_stage_v_common_v13 import NS, seeds
from .native_stage_v_evaluator_v13 import posterior, metrics
from ..generator import v13 as g


def optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001,
                            betas=(.9, .999), eps=1e-8, amsgrad=False, foreach=False, fused=False)


def shuffle(cell, corpus, init, n, epoch):
    if n not in NS or not 0 <= epoch < 100:
        raise ValueError('outside frozen training protocol')
    namespace = f'g13/stage_v/shuffle/{cell}/{corpus}/{init}/{n}/{epoch}'
    generator = torch.Generator(device='cpu').manual_seed(derive(namespace))
    return torch.randperm(n, generator=generator, device='cpu')


def environment():
    import platform
    if not torch.cuda.is_available():
        raise ValueError('certified GPU required; no CPU fallback')
    name = torch.cuda.get_device_name(0)
    if name != 'NVIDIA GeForce RTX 4080 SUPER':
        raise ValueError('uncertified GPU: ' + name)
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                cuda_runtime=torch.version.cuda, gpu=name, dtype='float64',
                deterministic=True, tf32=False, cublas_workspace=':4096:8', threads=1)


def mechanism(model, cache):
    # Fixed public q, only topology differs. Temporal order intervention is sequence-only.
    q = g.support()[0]
    other = g.support().index(replace(q, topology='join_d'))
    with torch.no_grad():
        first = select(cache, [0])
        original = forward(model, first)
        result = {}
        if model.family != 'tabular':
            result['topology_delta'] = float((forward(model, select(cache, [other]))-original).abs().max())
        if model.family == 'temporal':
            changed = dict(first)
            changed['sequences'] = first['sequences'].clone()
            changed['sequences'][:, :, [1, 2]] = changed['sequences'][:, :, [2, 1]]
            result['sequence_order_delta'] = float((forward(model, changed)-original).abs().max())
            result['sequence_neutralization_delta'] = float((forward(model, first, neutralize_sequence=True)-original).abs().max())
        if any(not torch.isfinite(torch.tensor(v)) for v in result.values()):
            raise ValueError('nonfinite mechanism diagnostic')
        return result


def train(identity, rows, cache, *, epoch_callback=None):
    """One fresh official run. Caller persists failures; no internal retry."""
    env = environment()
    seed = seeds()[f"g13/stage_v/init/{identity['init']}"]['seed']
    model = build_model(identity['family'], seed, 'cuda')
    opt = optimizer(model)
    n = identity['N']
    if n not in NS or len(rows) != 8192:
        raise ValueError('invalid frozen corpus prefix')
    inputs = torch.tensor([r['public_support_index'] for r in rows[:n]], device='cuda')
    labels = torch.tensor([r['y'] for r in rows[:n]], device='cuda', dtype=torch.float64)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.monotonic()
    model.train()
    for epoch in range(100):
        order = shuffle(identity['cell'], identity['corpus'], identity['init'], n, epoch).to('cuda')
        total_loss = torch.zeros((), dtype=torch.float64, device='cuda')
        finite = torch.ones((), dtype=torch.bool, device='cuda')
        for batch in order.split(32):
            opt.zero_grad(set_to_none=True)
            logits = forward(model, select(cache, inputs[batch]))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels[batch], reduction='mean')
            loss.backward()
            # Device-side accumulation checks every minibatch without a host sync per parameter.
            finite.logical_and_(torch.isfinite(loss))
            gradients = [p.grad for p in model.parameters()]
            if any(gradient is None for gradient in gradients):
                raise ValueError('missing parameter gradient')
            finite.logical_and_(torch.isfinite(torch.cat([g.reshape(-1) for g in gradients])).all())
            opt.step()
            total_loss += loss.detach()*len(batch)
        if not bool(finite):
            raise ValueError(f'nonfinite loss/gradient at epoch {epoch}')
        if epoch_callback:
            epoch_callback(epoch+1)
    final_loss = float(total_loss/n)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError('nonfinite final parameter')
    model.eval()
    evaluated, predictions = {}, {}
    with torch.no_grad():
        for partition in ('validation', 'test'):
            exact = posterior(identity['cell'], partition)
            qhat = torch.sigmoid(forward(model, select(cache, list(exact)))).cpu().tolist()
            predictions[partition] = dict(support_indices=list(exact), probabilities=qhat)
            evaluated[partition] = metrics(qhat, list(exact.values()), identity['cell'].split('-')[1])
    diagnostics = mechanism(model, cache)
    state_hash = parameter_digest(model)
    torch.cuda.synchronize()
    return dict(initialization_seed=str(seed), parameter_count=sum(p.numel() for p in model.parameters()),
                device_metadata=env, final_train_loss=final_loss, evaluation=evaluated,
                predictions=predictions, mechanism_diagnostics=diagnostics,
                final_state_dict_sha256=state_hash, final_parameters_finite=True,
                optimizer_steps=100*n//32, example_presentations=100*n,
                runtime_seconds=time.monotonic()-start,
                max_allocated_gpu_memory_bytes=torch.cuda.max_memory_allocated())
