"""Read-only optimizer-step hooks, narrow detailed windows plus scalar census."""
import copy,hashlib,json
from pathlib import Path
import torch
from capture_trace import pairs
from adamw_step_decomposition import boundary_analysis,reconstruct

def digest(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
def norms(t):
    u=t.reshape(t.shape[0],-1)
    return dict(l2=float(t.norm()),per_unit_l2=u.norm(dim=1).cpu().tolist(),min=float(t.min()),max=float(t.max()))

@torch.no_grad()
def snapshot(part,opt):
    out={}
    for k,p in part.named_parameters():
        s=opt.state.get(p,{})
        out[k]=dict(parameter=p.detach().clone(),gradient=p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p),step=s.get('step',torch.tensor(0.)).clone(),exp_avg=s.get('exp_avg',torch.zeros_like(p)).clone(),exp_avg_sq=s.get('exp_avg_sq',torch.zeros_like(p)).clone())
    return out

def summarize(s,group):
    out={};b1,b2=group['betas']
    for k,v in s.items():
        step=float(v['step']);out[k]=dict(parameter=norms(v['parameter']),gradient=norms(v['gradient']),exact_zero_gradient_units=int((v['gradient'].reshape(v['gradient'].shape[0],-1)==0).all(1).sum()),parameter_sha256=digest(v['parameter']),step=step,exp_avg=norms(v['exp_avg']),exp_avg_sq=norms(v['exp_avg_sq']),bias_correction1=1-b1**step,bias_correction2=1-b2**step)
        if k=='bias':out[k]['values']=v['parameter'].cpu().tolist()
    return out

class Capture:
    def __init__(self,model,opt,cache,identity,case,out):
        self.model=model;self.opt=opt;self.part=model.parts['content'];self.identity=identity;self.case=case;self.out=out
        self.mask=cache['mask'].bool();self.x=torch.where(self.mask.unsqueeze(-1),cache['x'],torch.zeros_like(cache['x']))
        self.group=opt.param_groups[0];assert all(any(p is q for q in self.group['params']) for p in self.part.parameters())
        assert len(opt.param_groups)==1 and not self.group['foreach'] and not self.group['fused']
        self.original_step=opt.step;self.initialized=False;self.first_death=None;self.order=None;self.order_epoch=None
        opt.step=self.step
    @torch.no_grad()
    def geometry(self,full=False):
        z=self.part(self.x);valid=z[self.mask];a=torch.relu(z)*self.mask.unsqueeze(-1)
        r=dict(max_preactivation=float(valid.max()),min_preactivation=float(valid.min()),dead=bool((valid<=0).all()),post_relu_all_zero=bool((a==0).all()))
        assert r['dead']==r['post_relu_all_zero']
        if full:
            l,rr=pairs();d=(a[l]-a[rr]).reshape(64,-1).norm(dim=1)
            r.update(positive_fraction=float((valid>0).double().mean()),live_units=int((valid>0).any(0).sum()),per_unit_max=valid.max(0).values.cpu().tolist(),per_unit_positive_fraction=(valid>0).double().mean(0).cpu().tolist(),post_relu_l2=float(a.norm()),post_relu_active_variance=float(torch.relu(valid).var(unbiased=False)),h_pair_l2_mean=float(d.mean()),h_pair_l2_max=float(d.max()))
        return r
    def initialize(self,completed):
        # Reconcile diagnostics to the durable official-format epoch checkpoint.
        for name in ['steps.jsonl','detail.jsonl']:
            p=self.out/name
            if p.exists():
                rows=[json.loads(l) for l in p.read_text().splitlines() if l.strip()]
                rows=[r for r in rows if r['global_step']<=completed]
                p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
                if name=='steps.jsonl':
                    ds=[r['global_step'] for r in rows if r['after']['dead']];self.first_death=min(ds) if ds else None
        if (self.out/'boundary.json').exists():
            b=json.loads((self.out/'boundary.json').read_text())
            if b['global_step']>completed:(self.out/'boundary.json').unlink()
        self.initialized=True
    @torch.no_grad()
    def step(self,*args,**kwargs):
        completed=int(self.opt.state.get(self.part.weight,{}).get('step',0));s=completed+1
        if not self.initialized:self.initialize(completed)
        spe=self.identity['N']//32;epoch=(s-1)//spe+1;batch=(s-1)%spe+1
        window=(47<=epoch<=50) if self.case=='D1' else (9<=epoch<=12 if self.case in ['D3','D4'] else False)
        detailed=window or s% spe==0 or s==1 or s==6400 or (self.case=='D2' and s%32==0)
        before=self.geometry(detailed)
        pre=snapshot(self.part,self.opt) if detailed else None
        full_pre=dict(model=copy.deepcopy(self.model.state_dict()),optimizer=copy.deepcopy(self.opt.state_dict())) if window and self.case in ['D1','D3'] else None
        self.original_step(*args,**kwargs)
        after=self.geometry(detailed)
        row=dict(case=self.case,run_id=self.identity,global_step=s,epoch=epoch,batch=batch,before=before,after=after)
        with (self.out/'steps.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        if detailed:
            post=snapshot(self.part,self.opt);detail=dict(row,parameters_before=summarize(pre,self.group),parameters_after=summarize(post,self.group))
            # Compare immediate adaptive/decay margins also in the matched control.
            dd={k:pre[k]['parameter']*(1-self.group['lr']*self.group['weight_decay'])-pre[k]['parameter'] for k in pre}
            da={k:post[k]['parameter']-pre[k]['parameter']-dd[k] for k in pre}
            calc=lambda v:torch.nn.functional.linear(self.x,v['weight'],v['bias'])[self.mask]
            detail['update_geometry']=dict(decay_max_abs=float(calc(dd).abs().max()),adaptive_max_abs=float(calc(da).abs().max()))
            with (self.out/'detail.jsonl').open('a') as f:f.write(json.dumps(detail)+'\n')
            if s==6400:torch.save(dict(content=post,geometry=after),self.out/'step6400.pt')
            if self.case=='D4' and (self.out.parent/'D3/boundary.json').exists():
                target=json.loads((self.out.parent/'D3/boundary.json').read_text())['global_step']
                if s==target:torch.save(dict(before=pre,after=post,geometry_before=before,geometry_after=after),self.out/'matched_boundary.pt')
        if after['dead'] and self.first_death is None:
            self.first_death=s
            if before['dead'] or not window or full_pre is None:raise RuntimeError(f'First death outside detailed bracket: {self.case}/{s}; expand minimally before interpretation')
            result,dz=boundary_analysis(pre,post,self.group,self.x,self.mask)
            from run_diagnostics import training,save
            order=training.shuffle(self.identity['cell'],self.identity['corpus'],self.identity['init'],self.identity['N'],epoch-1)
            indices=order.split(32)[batch-1].tolist()
            torch.save(dict(**full_pre,content_before=pre,content_after=post,minibatch_row_indices=indices,identity=self.identity,global_step=s),self.out/'boundary_state.pt')
            torch.save(dz,self.out/'boundary_deltas.pt')
            result.update(case=self.case,run_id=self.identity,global_step=s,epoch=epoch,batch=batch,optimizer_group={k:v for k,v in self.group.items() if k!='params'},minibatch_row_indices=indices)
            save(self.out/'boundary.json',result)
            print('BOUNDARY',self.case,s,epoch,batch,result['attribution_status'],flush=True)
