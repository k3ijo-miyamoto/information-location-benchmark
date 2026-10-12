"""Frozen-copy one-step arithmetic; never mutates the training optimizer/model."""
import copy
import torch
ATOL, RTOL = 1e-14, 1e-12

@torch.no_grad()
def reconstruct(s, group, *, no_decay=False, zero_gradient=False, zero_moments=False):
    p=s['parameter'].clone();g=s['gradient'].clone();m=s['exp_avg'].clone();v=s['exp_avg_sq'].clone()
    if zero_gradient:g.zero_()
    if zero_moments:m.zero_();v.zero_()
    t=s['step'].clone();t.add_(1);step=float(t)
    lr,b1,b2,eps=group['lr'],*group['betas'],group['eps']
    if not no_decay:p.mul_(1-lr*group['weight_decay'])
    decayed=p.clone()
    m.lerp_(g,1-b1);v.mul_(b2).addcmul_(g,g,value=1-b2)
    denom=(v.sqrt()/(1-b2**step)**.5).add_(eps)
    p.addcdiv_(m,denom,value=-lr/(1-b1**step))
    return dict(parameter=p,exp_avg=m,exp_avg_sq=v,step=t,decayed=decayed)

@torch.no_grad()
def pytorch_copy_step(s,group,**flags):
    p=torch.nn.Parameter(s['parameter'].clone());g=s['gradient'].clone()
    if flags.get('zero_gradient'):g.zero_()
    opt=torch.optim.AdamW([p],lr=group['lr'],betas=group['betas'],eps=group['eps'],weight_decay=0 if flags.get('no_decay') else group['weight_decay'],foreach=False,fused=False,amsgrad=False)
    st={k:s[k].clone() for k in ['step','exp_avg','exp_avg_sq']}
    if flags.get('zero_moments'):st['exp_avg'].zero_();st['exp_avg_sq'].zero_()
    opt.state[p]=st;p.grad=g;opt.step()
    return dict(parameter=p.detach().clone(),**opt.state[p])

def parity(a,b):
    fields={k:dict(bitwise=torch.equal(a[k],b[k]),max_abs_error=float((a[k]-b[k]).abs().max()),within_tolerance=bool(torch.allclose(a[k],b[k],atol=ATOL,rtol=RTOL))) for k in ['parameter','exp_avg','exp_avg_sq','step']}
    return dict(atol=ATOL,rtol=RTOL,fields=fields,passed=all(x['within_tolerance'] for x in fields.values()))

@torch.no_grad()
def boundary_analysis(pre,post,group,x,mask):
    flags={'CF0':{},'CF1':{'no_decay':True},'CF2':{'zero_gradient':True},'CF3':{'zero_moments':True},'moment_only_no_decay':{'zero_gradient':True,'no_decay':True},'current_only_no_decay':{'zero_moments':True,'no_decay':True}}
    def z(values):return torch.nn.functional.linear(x,values['weight'],values['bias'])
    original={k:s['parameter'] for k,s in pre.items()};actual={k:s['parameter'] for k,s in post.items()}
    def measure(values):
        zz=z(values);valid=zz[mask];return dict(max_preactivation=float(valid.max()),dead=bool((valid<=0).all()),post_relu_all_zero=bool((torch.relu(valid)==0).all()))
    parity_results={};cf={};reconstructed={}
    for label,f in flags.items():
        r={k:reconstruct(s,group,**f) for k,s in pre.items()};reconstructed[label]=r
        parity_results[label]={k:parity(r[k],pytorch_copy_step(s,group,**f)) for k,s in pre.items()}
        if label=='CF0':parity_results['actual_optimizer']={k:parity(r[k],post[k]) for k in r}
        cf[label]=measure({k:s['parameter'] for k,s in r.items()})
    assert all(v['passed'] for r in parity_results.values() for v in r.values()),'AdamW parity failed; no attribution'
    assert all(v['fields']['parameter']['bitwise'] for v in parity_results['actual_optimizer'].values()),'CF0 not bitwise exact; stop boundary attribution'
    cf['CF4']=measure(original)
    decayed={k:s['decayed'] for k,s in reconstructed['CF0'].items()};cf['decay_only']=measure(decayed)
    za=z(original);zb=z(actual)
    dd={k:decayed[k]-original[k] for k in original};da={k:actual[k]-decayed[k] for k in original}
    dzd=z(dd);dza=z(da);delta=zb-za;residual=delta-dzd-dza
    locations=(za>0)&mask.unsqueeze(-1);survivors=[]
    for state,record,unit in locations.nonzero().cpu().tolist():
        key=(state,record,unit);survivors.append(dict(state=state,record=record,unit=unit,before=float(za[key]),adaptive=float(dza[key]),decay=float(dzd[key]),total_shift=float(delta[key]),after=float(zb[key]),rounding_residual=float(residual[key])))
    suff=[]
    if cf['decay_only']['dead']:suff.append('DECAY_SUFFICIENT')
    if cf['CF2']['dead'] and cf['moment_only_no_decay']['dead']:suff.append('MOMENT_HISTORY_SUFFICIENT')
    if cf['CF3']['dead'] and cf['current_only_no_decay']['dead']:suff.append('CURRENT_GRADIENT_SUFFICIENT')
    assert cf['CF0']['dead'] and not cf['CF4']['dead']
    if len(suff)==1:status=suff[0]
    elif len(suff)>1:status='UNRESOLVED'
    elif not cf['CF2']['dead'] and not cf['CF3']['dead']:status='COMBINED_UPDATE_REQUIRED'
    else:status='UNRESOLVED'
    detailed=dict(z_before=za.cpu(),z_after=zb.cpu(),delta_z_total=delta.cpu(),delta_z_adaptive=dza.cpu(),delta_z_decay=dzd.cpu(),rounding_residual=residual.cpu(),active_mask=mask.cpu())
    return dict(counterfactuals=cf,parity=parity_results,supported_sufficiency_labels=suff,attribution_status=status,last_surviving_units=sorted(set(r['unit'] for r in survivors)),last_positive_records=survivors,max_abs_additivity_residual=float(residual[mask].abs().max()),max_abs_adaptive=float(dza[mask].abs().max()),max_abs_decay=float(dzd[mask].abs().max())),detailed
