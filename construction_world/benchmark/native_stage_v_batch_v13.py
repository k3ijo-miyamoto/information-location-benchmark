"""Behavior-preserving batched path for frozen NativeModel parameters/equations."""
import torch
from .native_tensor_v13 import prepare, reconstruct
from . import native_inputs_v13 as adapters
from ..generator.v13_projection import normalize


def stack(rows):
    if not rows or any(set(row) != set(rows[0]) for row in rows):
        raise ValueError('nonempty homogeneous tensor rows required')
    return {key: torch.stack([row[key] for row in rows]) for key in rows[0]}


def select(cache, indices):
    return {key: value[indices] for key, value in cache.items()}


def templates(publics):
    result = {}
    for family in ('tabular', 'static', 'temporal'):
        rows = []
        for public in publics:
            row = prepare(adapters.encode(public, family))
            if reconstruct(row) != normalize(public):
                raise ValueError('template fact roundtrip mismatch')
            rows.append(row)
        result[family] = stack(rows)
    # No second definition of Static-size-control facts.
    result['static_size_control'] = result['static']
    return result


def forward(model, batch, *, neutralize_sequence=False):
    mask = batch['mask'].unsqueeze(-1).bool()
    x = torch.where(mask, batch['x'], torch.zeros_like(batch['x']))
    parts = model.parts
    if model.family == 'tabular':
        endpoints = torch.where(mask, batch['endpoints'], torch.zeros_like(batch['endpoints']))
        content = torch.relu(parts['content'](x))
        refs = torch.relu(parts['reference_projection'](parts['endpoints'](endpoints).flatten(start_dim=2)))
        records = torch.cat((content, refs), dim=2)*mask
        fused = torch.relu(parts['fusion'](records.flatten(start_dim=1)))
        fused = torch.relu(parts['hidden'](fused))
    else:
        h = torch.relu(parts['content'](x))*mask
        size, nodes, width = h.shape
        offset = torch.arange(size, device=h.device)[:, None]*nodes
        source, destination, types = batch['edges'].unbind(-1)
        source = (source+offset).flatten()
        destination = (destination+offset).flatten()
        types = types.flatten()
        degree = h.new_zeros((size*nodes, 1)).index_add(0, destination, h.new_ones((len(destination), 1)))
        for layer in parts['messages']:
            flat = h.reshape(-1, width)
            messages = torch.relu(layer['message'](torch.cat((flat[source], parts['edge_types'](types)), dim=1)))
            aggregate = torch.zeros_like(flat).index_add(0, destination, messages)
            h = torch.relu(layer['self'](h)+(aggregate/degree.clamp_min(1)).reshape_as(h))*mask
        pieces = [h[torch.arange(size, device=h.device), batch['target']],
                  h.sum(1)/batch['mask'].sum(1, keepdim=True)]
        if model.family == 'temporal':
            if neutralize_sequence:
                history = torch.zeros_like(pieces[0])
            else:
                sequence = batch['sequences']+offset.unsqueeze(-1)
                state = h.new_zeros((size*2, width))
                flat = h.reshape(-1, width)
                for step in range(7):
                    state = parts['gru'](flat[sequence[:, :, step].flatten()], state)
                history = state.reshape(size, 2, width).mean(1)
            pieces.append(history)
        fused = torch.relu(parts['fusion'](torch.cat(pieces, dim=1)))
    return parts['output'](fused).squeeze(-1)
