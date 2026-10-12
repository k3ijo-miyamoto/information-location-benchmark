"""Read-only measurements and exact-state utilities for one fixed late-LR arm."""
import hashlib,io,json,os,tempfile
from pathlib import Path
import torch
from capture_trace import pairs,capture_epoch
from capture_optimizer_boundary import snapshot,summarize

def tree_digest(value):
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            h.update(str((str(x.dtype),tuple(x.shape))).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes())
        elif isinstance(x,dict):
            h.update(b'dict')
            for k in sorted(x,key=lambda a:(type(a).__name__,repr(a))):visit(k);visit(x[k])
        elif isinstance(x,(list,tuple)):
            h.update(type(x).__name__.encode())
            for v in x:visit(v)
        else:h.update((type(x).__name__+':'+repr(x)).encode())
    visit(value);return h.hexdigest()

def exact_tree(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a.detach().cpu(),b.detach().cpu())
    if type(a)!=type(b):return False
    if isinstance(a,dict):return a.keys()==b.keys() and all(exact_tree(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(exact_tree(x,y) for x,y in zip(a,b))
    return a==b

def rng_state():return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all())

@torch.no_grad()
def geometry(model,batch):
    mask=batch['mask'].bool();x=torch.where(mask.unsqueeze(-1),batch['x'],torch.zeros_like(batch['x']));z=model.parts['content'](x);valid=z[mask];a=torch.relu(z)*mask.unsqueeze(-1)
    l,r=pairs();dist=(a[l]-a[r]).reshape(64,-1).norm(dim=1);mx=valid.max(0).values
    dead=bool((valid<=0).all());zero=bool((a==0).all());assert dead==zero
    return dict(max_preactivation=float(valid.max()),min_preactivation=float(valid.min()),dead=dead,post_relu_all_zero=zero,live_units=int((mx>0).sum()),live_unit_indices=(mx>0).nonzero().flatten().cpu().tolist(),positive_fraction=float((valid>0).double().mean()),post_relu_l2=float(a.norm()),h_pair_l2_mean=float(dist.mean()),h_pair_l2_max=float(dist.max()))

@torch.no_grad()
def update_geometry(before,model,batch,lr,wd):
    mask=batch['mask'].bool();x=torch.where(mask.unsqueeze(-1),batch['x'],torch.zeros_like(batch['x']));after=dict(model.parts['content'].named_parameters())
    dd={k:v*(1-lr*wd)-v for k,v in before.items()};da={k:after[k]-before[k]-dd[k] for k in before}
    z=lambda v:torch.nn.functional.linear(x,v['weight'],v['bias'])
    za=z(before);zb=z(after);idx=za.masked_fill(~mask.unsqueeze(-1),-torch.inf).reshape(-1).argmax();shape=za.shape
    index=(int(idx)//(shape[1]*shape[2]),(int(idx)//shape[2])%shape[1],int(idx)%shape[2])
    ad,de=z(da),z(dd)
    return dict(adaptive_max_abs=float(ad[mask].abs().max()),decay_max_abs=float(de[mask].abs().max()),last_max_location=list(index),last_max_before=float(za[index]),last_max_after=float(zb[index]),adaptive_at_last_max=float(ad[index]),decay_at_last_max=float(de[index]))

def epoch_measure(model,opt,cache,rows,identity,epoch):
    r=capture_epoch(model,cache,rows[:identity['N']],identity,epoch)
    r['geometry']=geometry(model,cache);r['content_optimizer']=summarize(snapshot(model.parts['content'],opt),opt.param_groups[0]);return r

def atomic_checkpoint(path,state):
    for name in ['steps.jsonl','epochs.jsonl']:
        trace=path.parent/name
        if trace.exists():
            with trace.open('rb') as f:os.fsync(f.fileno())
    b=io.BytesIO();torch.save(state,b);payload=b.getvalue();data=b'G13-LATE-LR-01\n'+hashlib.sha256(payload).hexdigest().encode()+b'\n'+payload
    fd,name=tempfile.mkstemp(prefix='.checkpoint-',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        os.replace(name,path)
        fd=os.open(path.parent,os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if os.path.exists(name):os.unlink(name)

def load_checkpoint(path):
    magic,h,payload=path.read_bytes().split(b'\n',2);assert magic==b'G13-LATE-LR-01' and hashlib.sha256(payload).hexdigest().encode()==h
    return torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)

def trim_trace(path,completed):
    if not path.exists():return
    rows=[]
    for line in path.read_text().splitlines():
        try:r=json.loads(line)
        except json.JSONDecodeError:break # interrupted final write, never authoritative
        if r['global_step']<=completed:rows.append(r)
    path.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
