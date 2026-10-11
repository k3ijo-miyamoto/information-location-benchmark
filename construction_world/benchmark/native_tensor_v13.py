"""Lossless numeric learner boundary for frozen v1.3 public adapters."""
import math

import torch
from torch.nn import functional as F

from . import native_inputs_v13 as adapter
from ..generator.v13_projection import TABLES

SLOTS, FIELDS, CONTENT = 128, 73, 2575


def prepare(encoded):
    family = encoded.get('family')
    graph = family in ('static','temporal')
    allowed = {'family','mask','tables','states','values','as_of'} | ({'reference_edges'} if graph else {'endpoints'})
    if family == 'temporal': allowed.add('sequences')
    if family not in ('tabular','static','temporal') or set(encoded) != allowed:
        raise ValueError('unknown family or field')
    for key in ('mask','tables','states','values'):
        if len(encoded[key]) != SLOTS: raise ValueError('slot overflow/shape')
    if any(m not in (0,1) for m in encoded['mask']) or not any(encoded['mask']):
        raise ValueError('invalid active mask')
    for key in ('states','values'):
        if any(len(row) != FIELDS for row in encoded[key]): raise ValueError('primitive shape')
    if encoded['as_of'] != 7200: raise ValueError('invalid public cutoff')
    active = [i for i,m in enumerate(encoded['mask']) if m]
    columns, categories, permitted = adapter.layout()
    if (len(columns),len(categories),len(TABLES)) != (73,26,18): raise ValueError('frozen schema dimensions changed')
    states = torch.zeros((SLOTS,FIELDS),dtype=torch.long)
    nums = torch.zeros((SLOTS,FIELDS),dtype=torch.float64)
    cats = torch.zeros((SLOTS,FIELDS,26),dtype=torch.float64)
    tables = torch.zeros((SLOTS,18),dtype=torch.float64)
    counts = {}
    for i in active:
        t = encoded['tables'][i]
        if type(t) is not int or not 0 <= t < 18: raise ValueError('unknown table')
        table = TABLES[t]; counts[table] = counts.get(table,0)+1
        if counts[table] > adapter.CAPS.get(table,0): raise ValueError('table overflow')
        tables[i,t] = 1
        for j,(state,value) in enumerate(zip(encoded['states'][i],encoded['values'][i])):
            if type(state) is not int or state not in range(8): raise ValueError('unknown state')
            if state and columns[j] not in permitted[table]: raise ValueError('unknown primitive field')
            if not isinstance(value,(int,float)) or not math.isfinite(value): raise ValueError('nonfinite primitive')
            states[i,j] = state
            if state == 1: nums[i,j] = value/7200
            elif state == 3:
                if value not in (0,1): raise ValueError('invalid boolean')
                nums[i,j] = value
            elif state == 2:
                if type(value) is not int or not 0 <= value < 26: raise ValueError('unknown category')
                cats[i,j,value] = 1
    refs = []
    if graph:
        refs = encoded['reference_edges']
    else:
        matrix = encoded['endpoints']
        if len(matrix) != SLOTS or any(len(r)!=FIELDS for r in matrix): raise ValueError('endpoint shape')
        for i in active:
            for j,end in enumerate(matrix[i]):
                if type(end) is not int or not 0 <= end <= SLOTS: raise ValueError('invalid endpoint')
                if end: refs.append([i,j,end-1])
    seen = set(); endpoints = torch.zeros((SLOTS,FIELDS),dtype=torch.long)
    for row in refs:
        if len(row)!=3 or any(type(v) is not int for v in row): raise ValueError('invalid reference')
        i,j,target = row
        if i not in active or target not in active or not 0 <= j < FIELDS: raise ValueError('reference to padding/unknown')
        if (i,j) in seen or states[i,j] != 7: raise ValueError('duplicate/untyped reference')
        seen.add((i,j)); endpoints[i,j] = target+1
    if seen != {(i,j) for i in active for j in range(FIELDS) if states[i,j]==7}:
        raise ValueError('missing typed reference')
    target_field = columns.index(('target_activity_ids',0))
    targets = {int(endpoints[i,target_field])-1 for i in active
               if encoded['tables'][i] == TABLES.index('risk_assessments') and endpoints[i,target_field]}
    if len(targets)!=1: raise ValueError('unique public target required')
    mask = torch.tensor(encoded['mask'],dtype=torch.float64)
    content = torch.cat((nums[:,:,None],F.one_hot(states,8).to(torch.float64),cats),dim=2).reshape(SLOTS,-1)
    x = torch.cat((content,tables,mask[:,None],torch.ones((SLOTS,1),dtype=torch.float64)),dim=1)
    x = torch.where(mask[:,None].bool(),x,torch.zeros_like(x))
    result = dict(x=x,mask=mask,target=torch.tensor(next(iter(targets)),dtype=torch.long))
    if graph:
        edges = [[i,target,j] for i,j,target in refs]+[[target,i,j+FIELDS] for i,j,target in refs]
        result['edges'] = torch.tensor(sorted(edges),dtype=torch.long).reshape(-1,3)
    else: result['endpoints'] = endpoints
    if family == 'temporal':
        chains = encoded['sequences']
        if len(chains)!=2 or any(len(c)!=7 for c in chains): raise ValueError('two seven-record streams required')
        if any(type(i) is not int or i not in active for c in chains for i in c): raise ValueError('sequence to padding/unknown')
        if any(len(set(c))!=7 for c in chains): raise ValueError('duplicate sequence member')
        result['sequences'] = torch.tensor(chains,dtype=torch.long)
    return result


def reconstruct(batch):
    """Decode facts from learner tensors only; no encoded/raw sidecar."""
    x, mask = batch['x'].detach().cpu(),batch['mask'].detach().cpu()
    fields = x[:,:73*35].reshape(128,73,35)
    states = fields[:,:,1:9].argmax(2)
    values = torch.zeros((128,73),dtype=torch.float64)
    for i in range(128):
        if not mask[i]: continue
        for j in range(73):
            s = int(states[i,j])
            if s == 1:
                value = float(fields[i,j,0]*7200)
                values[i,j] = round(value) if abs(value-round(value))<1e-9 else value
            elif s == 3: values[i,j] = fields[i,j,0]
            elif s == 2: values[i,j] = fields[i,j,9:].argmax()
    raw_values = values.tolist()
    for i in range(128):
        for j in range(73):
            if states[i,j] == 2: raw_values[i][j] = int(raw_values[i][j])
    encoded = dict(mask=[int(m) for m in mask.tolist()],states=states.tolist(),values=raw_values,
                   tables=x[:,73*35:73*35+18].argmax(1).tolist(),as_of=7200)
    if 'endpoints' in batch:
        encoded.update(family='tabular',endpoints=batch['endpoints'].detach().cpu().tolist())
    else:
        edges = batch['edges'].detach().cpu().tolist()
        forward = [row for row in edges if row[2]<73]
        assert sorted([b,a,t+73] for a,b,t in forward) == sorted(row for row in edges if row[2]>=73)
        encoded.update(family='static',reference_edges=[[a,t,b] for a,b,t in forward])
    return adapter.reconstruct(encoded)


def move(batch,device):
    return {k:v.to(device) for k,v in batch.items()}


def permute_graph(batch):
    out = {k:v.clone() for k,v in batch.items()}
    order = torch.arange(127,-1,-1,device=batch['x'].device)
    for key in ('x','mask'): out[key] = out[key][order]
    out['target'] = 127-out['target']
    out['edges'] = out['edges'].flip(0)
    out['edges'][:,:2] = 127-out['edges'][:,:2]
    if 'sequences' in out: out['sequences'] = 127-out['sequences']
    return out
