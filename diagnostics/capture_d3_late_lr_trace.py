"""Read-only Temporal diagnostics at the prespecified cadence; source unchanged."""
import torch
from capture_late_lr_trace import epoch_measure as base_epoch_measure
from construction_world.benchmark import native_stage_v_training_v13 as training
from construction_world.benchmark.native_stage_v_batch_v13 import forward
@torch.no_grad()
def temporal_measure(model,cache,full_support=True):
    was_training=model.training;model.eval()
    result=dict(official_single_state=training.mechanism(model,cache))
    if full_support:
        logits=forward(model,cache);neutral=forward(model,cache,neutralize_sequence=True)
        changed=dict(cache);changed['sequences']=cache['sequences'].clone()
        changed['sequences'][:,:,[1,2]]=changed['sequences'][:,:,[2,1]]
        result['full_support']=dict(neutralization_logit_max=float((neutral-logits).abs().max()),order_swap_logit_max=float((forward(model,changed)-logits).abs().max()))
    model.train(was_training);return result

def epoch_measure(model,opt,cache,rows,identity,epoch):
    result=base_epoch_measure(model,opt,cache,rows,identity,epoch)
    result['official_mechanism']=temporal_measure(model,cache,full_support=False)['official_single_state']
    return result
