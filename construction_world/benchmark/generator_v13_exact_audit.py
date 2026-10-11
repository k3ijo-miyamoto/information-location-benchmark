"""Exact Generator1.3 audit. Deterministic support fixtures only; no ML or sampling."""
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
from datetime import timedelta
from decimal import Decimal, localcontext
from fractions import Fraction as F
from functools import lru_cache
from hashlib import sha256
import json
from math import comb, log2
from pathlib import Path

from ..generator import v13 as g
from ..generator.v13_exact import enumerate_atoms,buckets,posteriors,brier,accuracy
from ..generator.v13_projection import normalize,projections,key,ID_FIELDS,sequences
from . import native_inputs_v13 as adapters

ROOT=Path(__file__).resolve().parents[2]


def frac(value): return {'exact':str(value),'float':float(value)}
def digest(value): return sha256(key(value).encode()).hexdigest()


def entropy(groups):
    total=F(sum(m for m,p in groups.values()))
    float_value=0.0
    with localcontext() as ctx:
        ctx.prec=60
        reference=Decimal(0)
        for mass,positive in groups.values():
            p=positive/mass; weight=mass/total
            for q in (p,1-p):
                if q:
                    float_value-=float(weight*q)*log2(float(q))
                    dq=Decimal(q.numerator)/Decimal(q.denominator)
                    dw=Decimal(weight.numerator)/Decimal(weight.denominator)
                    reference-=dw*dq*dq.ln()/Decimal(2).ln()
    if abs(float_value-float(reference))>1e-12: raise AssertionError('entropy precision mismatch')
    return float_value


@lru_cache(maxsize=512)
def public_atom(q,ra,rb):
    data=g.build_world(q,ra,rb)  # build_world enforces unchanged v0.7 validation
    p=data.build_ai_view(g.ORIGIN+timedelta(minutes=120)) # revalidates Canonical
    events={e.event_id:e for e in data.events}
    realized={a:tuple(int((events[prefix+a].occurred_at-g.ORIGIN).total_seconds()/60) for prefix in ('START','END')) for a in 'ABCD'}
    original={w.activity_id:w for w in data.schedule_revisions[0].assignments}
    y=int((events['ENDD'].occurred_at-original['D'].planned_end).total_seconds()>=120*60)
    return p,normalize(p),realized,y


def reidentified(public):
    p=deepcopy(public)
    ids=[row[field] for table,field in ID_FIELDS.items() for row in p[table]]
    mapping={ident:'opaque-renamed-'+str(len(ids)-i) for i,ident in enumerate(ids)}
    def visit(v):
        if isinstance(v,dict): return {k:visit(x) for k,x in reversed(list(v.items()))}
        if isinstance(v,list): return [visit(x) for x in reversed(v)]
        return mapping.get(v,v) if isinstance(v,str) else v
    return visit(p)


def adapter_audit():
    checks=Counter(); support_rows=[]; p_hashes=set()
    for q in g.support():
        public,p,_,_=public_atom(q,0,0)
        assert normalize(reidentified(public))==p
        shifted=g.build_world(q,0,0,origin=g.ORIGIN+timedelta(days=123,hours=3)).build_ai_view(g.ORIGIN+timedelta(days=123,hours=3,minutes=120))
        assert normalize(shifted)==p
        checks['alpha_rename_source_order_absolute_origin']+=1
        encoded={family:adapters.encode(public,family) for family in ('tabular','static','temporal')}
        for family,e in encoded.items():
            assert adapters.reconstruct(e)==p
            assert adapters.encode(reidentified(public),family)==e
            padded=deepcopy(e)
            for i,active in enumerate(padded['mask']):
                if not active: padded['values'][i]=[999]*len(padded['values'][i])
            assert adapters.reconstruct(padded)==p
            if family!='tabular': assert adapters.reconstruct(adapters.permute_graph(e))==p
            checks[family+'_roundtrip_and_invariance']+=1
        static,temporal=encoded['static'],encoded['temporal']
        assert {k:v for k,v in static.items() if k!='family'}=={k:v for k,v in temporal.items() if k not in ('family','sequences')}
        assert all(len(chain)==7 for chain in temporal['sequences'])
        # Sequence order is reconstructible independently from decoded public facts.
        decoded=adapters.reconstruct(temporal)
        assert sequences(decoded)==sequences(p)
        checks['static_temporal_same_facts']+=1
        identity=digest(p); p_hashes.add(identity)
        support_rows.append(dict(coordinates=asdict(q),public_sha256=identity,mass='1/128',split=g.split(q)))
    assert len(p_hashes)==128
    return dict(status='pass',checks=dict(checks),record_slots=128,active_records=49,primitive_columns=len(adapters.layout()[0]),chain_lengths=[7,7],learner_runs=0),support_rows


def occupancy():
    results=[]
    for n in (250,500,1000,2000):
        groups=128*(1-(127/128)**n)
        bound=2*(7/8)**n+4*(15/16)**n+sum(comb(m,15)*(1-(m-15)/128)**n for m in (64,32,32))
        results.append(dict(total_future_draws=n,expected_observed_groups=groups,expected_duplicate_fraction=(n-groups)/n,split_infeasibility_upper_bound=min(1,bound)))
    return results


def audit_cell(regime,opportunity):
    rows=[]; canonical=0
    for atom in enumerate_atoms(regime,opportunity):
        q=atom.coordinates
        public,p,realized,serialized_y=public_atom(q,atom.ra,atom.rb)
        assert realized==atom.actual and serialized_y==atom.y
        # Compare actual serialized execution with the recurrence and original target.
        expected=(max(180*atom.ra,180*atom.rb) if q.topology=='merge_c' else max(180*atom.ra,180*atom.rb-q.c,0))
        assert atom.actual['D'][1]-g.baseline(q)['D'][1]==expected
        assert atom.y==(int(bool(atom.ra or atom.rb)) if q.topology=='merge_c' else atom.ra)
        assert p==public_atom(q,0,0)[1], 'post-cutoff realization leaked into P'
        canonical+=1
        rows.append(dict(q=q,mass=atom.mass,y=atom.y,keys=projections(p)))
    groupings={name:buckets(rows,lambda r,n=name:r['keys'][n]) for name in ('empty','Xc','XcT','XcH','full','P')}
    mass=sum(r['mass'] for r in rows);prevalence=sum(r['mass']*r['y'] for r in rows)
    bp={name:brier(groups) for name,groups in groupings.items()}
    en={name:entropy(groups) for name,groups in groupings.items()}
    ac={name:accuracy(groups) for name,groups in groupings.items()}
    pp=posteriors(groupings['P']); fp=posteriors(groupings['full'])
    coarse=posteriors(buckets(rows,lambda r:(r['q'].x,r['q'].topology,r['q'].h)))
    assert all(pp[r['keys']['P']]==fp[r['keys']['full']]==coarse[(r['q'].x,r['q'].topology,r['q'].h)] for r in rows)
    k=g.K[opportunity]
    gains={'current':bp['empty']-bp['Xc'],'topology':bp['Xc']-bp['XcT'],'history':bp['XcT']-bp['full'],'full':bp['empty']-bp['full'],'interaction':bp['XcT']-bp['full']-bp['Xc']+bp['XcH']}
    expected={'IS0':(81,0,0,0),'IS1':(0,81,0,0),'IS2':(0,0,81,0),'IS3':(0,49,32,16)}[regime]
    mi={'I_Y_Xc':en['empty']-en['Xc'],'I_Y_T_given_Xc':en['Xc']-en['XcT'],'I_Y_H_given_XcT':en['XcT']-en['full'],'I_Y_TH_given_Xc':en['Xc']-en['full']}
    failures=[]
    def gate(ok,reason):
        if not ok: failures.append(reason)
    gate(mass==1 and prevalence==F(1,2),'mass/prevalence')
    gate(bp['empty']==F(1,4) and gains['full']==81*k*k,'full Brier match')
    for name,coefficient in zip(('current','topology','history','interaction'),expected):
        gate(gains[name]==coefficient*k*k,'decomposition:'+name)
        if coefficient: gate(gains[name]>=F(1,10000),'positive risk gate:'+name)
    for name,coefficient in zip(('I_Y_Xc','I_Y_T_given_Xc','I_Y_H_given_XcT'),expected[:3]):
        gate(mi[name]>=1e-4 if coefficient else abs(mi[name])<=1e-12,'MI gate:'+name)
    gate(len(pp)==128 and all(m==F(1,128) for m,p in groupings['P'].values()),'public group count/mass')
    gate(all(F(1,4)<=p<=F(3,4) for p in pp.values()) and bp['full']>F(1,10),'non-oracle')
    if regime=='IS3':gate(bp['XcH']>bp['full'] and bp['XcT']>bp['full'],'single-structure gain')
    partitions={}
    for part in ('train','validation','test'):
        selected=[r for r in rows if g.split(r['q'])==part]
        part_mass=sum(r['mass'] for r in selected)
        normalized=[dict(r,mass=r['mass']/part_mass) for r in selected]
        groups=buckets(normalized,lambda r:r['keys']['P'])
        prevalence_part=sum(r['mass']*r['y'] for r in normalized)
        gain=F(1,4)-brier(groups)
        gate(prevalence_part==F(1,2) and gain==81*k*k,'partition matching:'+part)
        partitions[part]=dict(groups=len(groups),mass=str(part_mass),prevalence=str(prevalence_part),full_gain=str(gain))
    return dict(regime=regime,opportunity=opportunity,k=str(k),total_mass=str(mass),prevalence=frac(prevalence),support_group_count=len(pp),atomic_path_count=len(rows),posterior_min=frac(min(pp.values())),posterior_max=frac(max(pp.values())),mixed_label_group_mass='1',entropy_bits=en,mi_bits=mi,bayes_brier={n:frac(v) for n,v in bp.items()},bayes_log_loss_bits=en,bayes_accuracy={n:frac(v) for n,v in ac.items()},gains={n:frac(v) for n,v in gains.items()},nuisance_invariance_pass=True,full_public_posterior_equality_pass=True,canonical_validation_pass=True,canonical_weighted_paths_checked=canonical,adapter_parity_pass=True,split_support_pass=not any('partition' in f for f in failures),split=partitions,gate_failures=failures)


def audit():
    adapter,public_support=adapter_audit()
    groups=defaultdict(set);blocks=defaultdict(Counter)
    for r in public_support:
        q=g.Coordinates(**r['coordinates']);part=g.split(q)
        groups[part].add(r['public_sha256']);blocks[(q.x,q.topology,q.h)][part]+=1
    assert {p:len(ids) for p,ids in groups.items()}=={'train':64,'validation':32,'test':32}
    assert all(counts=={'train':8,'validation':4,'test':4} for counts in blocks.values())
    assert not any(groups[a]&groups[b] for a,b in (('train','test'),('train','validation'),('test','validation')))
    for part in groups:
        contexts=[r['coordinates'] for r in public_support if r['split']==part]
        for field,values in [('d',{30,45}),('c',{90,120,150,180}),('lead',{0,1})]: assert {r[field] for r in contexts}==values
    cells=[audit_cell(r,o) for r in g.REGIMES for o in g.K]
    count=sum(c['atomic_path_count'] for c in cells)
    assert count<=4096
    failures=[dict(regime=c['regime'],opportunity=c['opportunity'],failures=c['gate_failures']) for c in cells if c['gate_failures']]
    report=dict(version='generator13-exact-audit-0.1',decision='G13-B' if failures else 'G13-A',generator_version=g.VERSION,distribution=g.DISTRIBUTION,design=g.DESIGN,cells=cells,total_positive_mass_atomic_paths=count,atomic_path_bound=4096,unique_physical_coordinate_outcome_states=512,public_groups_per_cell=128,gate_failures=failures,actions=dict(sampled_benchmark_worlds=0,seeds_reserved=0,learner_runs=0,experts_contacted=0),next_task='design_and_freeze_learner_benchmark_and_sample_size_precision_plan')
    split=dict(status='pass',group_counts={p:len(v) for p,v in groups.items()},core_blocks=8,contexts_per_block=dict(train=8,validation=4,test=4),no_crossing=True,all_public_axis_values_per_partition=True,cell_results=[dict(regime=c['regime'],opportunity=c['opportunity'],partitions=c['split']) for c in cells],occupancy=occupancy())
    return report,dict(kind='exact_support_not_sampled_corpus',groups=public_support),adapter,split


def write_report(directory):
    directory=Path(directory)
    report,support,adapter,split=audit()
    for name,data in [('exact_population_audit',report),('public_support',support),('adapter_audit',adapter),('split_support_audit',split)]:
        (directory/f'generator_v1.3_{name}_v0.1.json').write_text(json.dumps(data,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=write_report(args.output)
    print(json.dumps({'decision':result['decision'],'paths':result['total_positive_mass_atomic_paths'],'failures':result['gate_failures']}))
    raise SystemExit(0 if result['decision']=='G13-A' else 1)
