"""Frozen v1.3 native architectures. Logits only; no optimizer or fitting API."""
import torch
from torch import nn


class NativeModel(nn.Module):
    def __init__(self,family):
        super().__init__()
        if family not in ('tabular','static','temporal','static_size_control'): raise ValueError('unknown family')
        self.family = family
        linear = lambda a,b: nn.Linear(a,b,dtype=torch.float64)
        h = 25 if family == 'static_size_control' else 24
        if family == 'tabular':
            self.parts = nn.ModuleDict(dict(content=linear(2575,8),endpoints=nn.Embedding(129,4,padding_idx=0,dtype=torch.float64),
                reference_projection=linear(292,8),fusion=linear(2048,21),hidden=linear(21,21),output=linear(21,1)))
        else:
            parts = dict(content=linear(2575,h),edge_types=nn.Embedding(146,4,dtype=torch.float64),
                messages=nn.ModuleList([nn.ModuleDict({'self':linear(h,h),'message':linear(h+4,h)}) for _ in range(2)]),
                fusion=linear((3 if family=='temporal' else 2)*h,h),output=linear(h,1))
            if family == 'temporal': parts['gru'] = nn.GRUCell(h,h,dtype=torch.float64)
            self.parts = nn.ModuleDict(parts)
        self.double()

    def component_counts(self):
        return {name:sum(p.numel() for p in module.parameters()) for name,module in self.parts.items()}

    def backbone(self,b):
        mask = b['mask'][:,None].bool()
        x = torch.where(mask,b['x'],torch.zeros_like(b['x']))
        h = torch.relu(self.parts['content'](x))*mask
        a,dst,types = b['edges'].unbind(1)
        for layer in self.parts['messages']:
            messages = torch.relu(layer['message'](torch.cat((h[a],self.parts['edge_types'](types)),dim=1)))
            aggregate = torch.zeros_like(h).index_add(0,dst,messages)
            degrees = h.new_zeros((128,1)).index_add(0,dst,h.new_ones((len(dst),1)))
            h = torch.relu(layer['self'](h)+aggregate/degrees.clamp_min(1))*mask
        return h

    def forward_one(self,b,neutralize_sequence=False):
        mask = b['mask'][:,None].bool()
        if self.family == 'tabular':
            x = torch.where(mask,b['x'],torch.zeros_like(b['x']))
            endpoints = torch.where(mask,b['endpoints'],torch.zeros_like(b['endpoints']))
            content = torch.relu(self.parts['content'](x))
            refs = torch.relu(self.parts['reference_projection'](self.parts['endpoints'](endpoints).flatten(start_dim=1)))
            records = torch.cat((content,refs),dim=1)*mask
            fused = torch.relu(self.parts['fusion'](records.flatten()))
            fused = torch.relu(self.parts['hidden'](fused))
        else:
            h = self.backbone(b)
            pieces = [h[b['target']],h.sum(0)/b['mask'].sum()]
            if self.family == 'temporal':
                if neutralize_sequence:
                    history = torch.zeros_like(pieces[0])
                else:
                    histories = []
                    for sequence in b['sequences']:
                        state = h.new_zeros(h.shape[1])
                        for index in sequence:
                            state = self.parts['gru'](h[index],state)
                        histories.append(state)
                    history = torch.stack(histories).mean(0)
                pieces.append(history)
            fused = torch.relu(self.parts['fusion'](torch.cat(pieces)))
        return self.parts['output'](fused).squeeze(-1)

    def forward(self,batch,neutralize_sequence=False):
        rows = [batch] if isinstance(batch,dict) else batch
        if not rows: raise ValueError('empty batch')
        return torch.stack([self.forward_one(b,neutralize_sequence) for b in rows])
