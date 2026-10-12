"""Read-only epoch diagnostics; no backward, RNG use, optimizer update or mutation."""
from dataclasses import replace
from functools import lru_cache
import torch
from construction_world.generator import v13 as g
from construction_world.benchmark.native_stage_v_batch_v13 import forward
from construction_world.benchmark.native_stage_v_evaluator_v13 import posterior,metrics

@lru_cache(None)
def pairs():
 support=g.support();left=[i for i,q in enumerate(support) if q.h==-1];right=[support.index(replace(support[i],h=1)) for i in left];return left,right

@torch.no_grad()
def capture_epoch(model,batch,rows,identity,epoch):
 was_training=model.training;model.eval();stages={};left,right=pairs();size=128
 def record(name,value):
  v=value.detach().reshape(size,-1);dist=(v[left]-v[right]).square().sum(1).sqrt()
  stages[name]=dict(h_pair_l2_mean=float(dist.mean()),h_pair_l2_max=float(dist.max()),h_pair_l2_min=float(dist.min()),zero_fraction=float((v==0).double().mean()),all_zero=bool((v==0).all()),variance=float(v.var(unbiased=False)),rms=float(v.square().mean().sqrt()))
  return value
 mask=batch['mask'].unsqueeze(-1).bool();x=torch.where(mask,batch['x'],torch.zeros_like(batch['x']));parts=model.parts
 record('raw_input',x)
 content_pre=record('content_pre',parts['content'](x));content=record('content_post',torch.relu(content_pre)*mask)
 if model.family=='tabular':
  endpoints=torch.where(mask,batch['endpoints'],torch.zeros_like(batch['endpoints']))
  refs=record('reference_post',torch.relu(parts['reference_projection'](parts['endpoints'](endpoints).flatten(start_dim=2))))
  fused_input=record('fusion_input',(torch.cat((content,refs),dim=2)*mask).flatten(start_dim=1))
 else:
  h=content;_,nodes,width=h.shape;offset=torch.arange(size,device=h.device)[:,None]*nodes
  source,dest,types=batch['edges'].unbind(-1);source=(source+offset).flatten();dest=(dest+offset).flatten();types=types.flatten()
  degree=h.new_zeros((size*nodes,1)).index_add(0,dest,h.new_ones((len(dest),1)))
  for num,layer in enumerate(parts['messages']):
   flat=h.reshape(-1,width);messages=torch.relu(layer['message'](torch.cat((flat[source],parts['edge_types'](types)),dim=1)));aggregate=torch.zeros_like(flat).index_add(0,dest,messages)
   h=record(f'gnn_{num+1}_post',torch.relu(layer['self'](h)+(aggregate/degree.clamp_min(1)).reshape_as(h))*mask)
  pieces=[record('target_readout',h[torch.arange(size,device=h.device),batch['target']]),record('global_readout',h.sum(1)/batch['mask'].sum(1,keepdim=True))]
  if model.family=='temporal':
   sequence=batch['sequences']+offset.unsqueeze(-1);state=h.new_zeros((size*2,width));flat=h.reshape(-1,width)
   for step in range(7):
    state=parts['gru'](flat[sequence[:,:,step].flatten()],state);record(f'gru_step_{step+1}',state.reshape(size,2,width))
   pieces.append(record('history_mean',state.reshape(size,2,width).mean(1)))
  fused_input=record('fusion_input',torch.cat(pieces,dim=1))
 fused=record('fusion_post',torch.relu(record('fusion_pre',parts['fusion'](fused_input))))
 if model.family=='tabular':fused=record('hidden_post',torch.relu(record('hidden_pre',parts['hidden'](fused))))
 logits=record('logit',parts['output'](fused)).squeeze(-1)
 official_logits=forward(model,batch)
 assert torch.equal(logits,official_logits),'Diagnostic forward differs from production batched forward'
 probabilities=torch.sigmoid(logits);preds=probabilities.cpu().tolist();evaluation={}
 for partition in ['validation','test']:
  exact=posterior(identity['cell'],partition);evaluation[partition]=metrics([preds[ix] for ix in exact],list(exact.values()),identity['cell'].split('-')[1])
 indices=torch.tensor([r['public_support_index'] for r in rows],device=logits.device);labels=torch.tensor([r['y'] for r in rows],dtype=torch.float64,device=logits.device)
 train_bce=float(torch.nn.functional.binary_cross_entropy_with_logits(logits[indices],labels))
 components={}
 for name,part in parts.items():
  pp=list(part.parameters());components[name]=dict(parameter_l2=float(sum(v.square().sum() for v in pp).sqrt()),last_minibatch_gradient_l2=float(sum(v.grad.square().sum() for v in pp if v.grad is not None).sqrt()))
 mechanism={}
 if model.family=='temporal':
  neutral=forward(model,batch,neutralize_sequence=True);changed=dict(batch);changed['sequences']=batch['sequences'].clone();changed['sequences'][:,:,[1,2]]=changed['sequences'][:,:,[2,1]]
  mechanism=dict(neutralization_logit_max=float((neutral-logits).abs().max()),order_swap_logit_max=float((forward(model,changed)-logits).abs().max()))
 model.train(was_training)
 return dict(epoch=epoch,diagnostic_forward_equal=True,train_bce_at_final_epoch_weights=train_bce,train_prevalence=sum(r['y'] for r in rows)/len(rows),prediction_mean=float(probabilities.mean()),prediction_range=float(probabilities.max()-probabilities.min()),probabilities=preds,evaluation=evaluation,stages=stages,components=components,mechanism=mechanism)
