"""Numeric, reversible public-record adapters; no learner modules or raw sidecar."""
from functools import lru_cache
from copy import deepcopy
from ..generator.v13 import Coordinates, ORIGIN, build_world
from ..generator.v13_projection import TABLES, ID_FIELDS, normalize, sequences
from datetime import timedelta

CAPS = {'entities':8,'relationships':24,'events':24,'event_links':8,'observations':24,'delivery_plans':2,'schedule_revisions':2,'risk_assessments':4,'blocking_conditions':2}
SLOTS=128


def leaves(value, path=()):
    if isinstance(value,dict) and value:
        for k,v in value.items(): yield from leaves(v,path+(k,))
    elif isinstance(value,list) and value:
        for i,v in enumerate(value): yield from leaves(v,path+(i,))
    else: yield path,value


@lru_cache(maxsize=1)
def layout():
    # Fixed schema fixture, not fitted vocabulary/data statistics. All declared
    # enum variants occur in this fixture (early low/high, root medium).
    p=normalize(build_world(Coordinates(),0,0).build_ai_view(ORIGIN+timedelta(minutes=120)))
    paths=set(); categories=set(); allowed={t:set() for t in TABLES}
    ids={r[ID_FIELDS[t]] for t in ID_FIELDS for r in p[t]}
    for t in TABLES:
        for row in p[t]:
            for path,value in leaves(row):
                paths.add(path); allowed[t].add(path)
                if isinstance(value,str) and value not in ids: categories.add(value)
    return tuple(sorted(paths,key=repr)),tuple(sorted(categories)),allowed


def encode(public, family):
    if family not in ('tabular','static','temporal'): raise ValueError('unknown family')
    p=normalize(public)
    columns,categories,allowed=layout()
    col={p:i for i,p in enumerate(columns)}
    rows=[(t,r) for t in TABLES for r in p[t]]
    if len(rows)>SLOTS or any(len(p[t])>CAPS.get(t,0) for t in TABLES): raise ValueError('record bound exceeded')
    ids={r[ID_FIELDS[t]]:i for i,(t,r) in enumerate(rows)}
    states=[[0]*len(columns) for _ in range(SLOTS)]
    values=[[0]*len(columns) for _ in range(SLOTS)]
    refs=[]
    for i,(table,row) in enumerate(rows):
        for path,value in leaves(row):
            if path not in allowed[table]: raise ValueError('unknown primitive path')
            j=col[path]
            if isinstance(value,str) and value in ids:
                state=7; number=0; refs.append([i,j,ids[value]])
            elif value is None: state=4; number=0
            elif value==[]: state=5; number=0
            elif value=={}: state=6; number=0
            elif isinstance(value,bool): state=3; number=int(value)
            elif isinstance(value,str):
                if value not in categories: raise ValueError('unknown category')
                state=2; number=categories.index(value)
            elif isinstance(value,(int,float)): state=1; number=value
            else: raise ValueError('unsupported value')
            states[i][j]=state; values[i][j]=number
    result=dict(family=family,mask=[1]*len(rows)+[0]*(SLOTS-len(rows)),tables=[TABLES.index(t) for t,r in rows]+[-1]*(SLOTS-len(rows)),states=states,values=values,as_of=p['as_of'])
    if family=='tabular':
        endpoints=[[0]*len(columns) for _ in range(SLOTS)]
        for i,j,target in refs: endpoints[i][j]=target+1
        result['endpoints']=endpoints
    else: result['reference_edges']=sorted(refs)
    if family=='temporal': result['sequences']=[[ids[ident] for ident in chain] for chain in sequences(p)]
    return result


def _assign(row,path,value):
    cursor=row
    for i,part in enumerate(path):
        last=i==len(path)-1
        if isinstance(cursor,list):
            while len(cursor)<=part: cursor.append(None)
        if last:
            cursor[part]=value
        else:
            next_container=[] if isinstance(path[i+1],int) else {}
            if isinstance(cursor,list):
                if cursor[part] is None: cursor[part]=next_container
            elif part not in cursor: cursor[part]=next_container
            cursor=cursor[part]


def reconstruct(encoded):
    columns,categories,_=layout()
    active=[i for i,m in enumerate(encoded['mask']) if m]
    if any(encoded['tables'][i] not in range(len(TABLES)) for i in active): raise ValueError('invalid table')
    if encoded['family']=='tabular':
        refs={(i,j):end-1 for i,row in enumerate(encoded['endpoints']) for j,end in enumerate(row) if end and i in active}
    else:
        edges=encoded['reference_edges']
        refs={(i,j):target for i,j,target in edges}
        if len(refs)!=len(edges): raise ValueError('duplicate typed reference')
    # Decoder assigns temporary IDs by physical tensor slot; canonical normalization
    # below removes node ordering. No raw record sidecar participates.
    result={t:[] for t in TABLES};result['as_of']=encoded['as_of']
    for i in active:
        row={}
        for j,state in enumerate(encoded['states'][i]):
            if not state: continue
            value=encoded['values'][i][j]
            if state==2: value=categories[value]
            elif state==3: value=bool(value)
            elif state==4: value=None
            elif state==5: value=[]
            elif state==6: value={}
            elif state==7:
                target=refs[(i,j)]
                if target not in active: raise ValueError('reference to padding')
                value='tensor:'+str(target)
            elif state!=1: raise ValueError('unknown state')
            _assign(row,columns[j],value)
        result[TABLES[encoded['tables'][i]]].append(row)
    # Re-express normalized seconds as absolute instants solely for the same
    # public normalizer; no fact is added or dropped.
    from ..generator.v13_projection import CLOCKS
    def dates(v,field=None):
        if isinstance(v,dict): return {k:dates(x,k) for k,x in v.items()}
        if isinstance(v,list): return [dates(x,field) for x in v]
        if field in CLOCKS and isinstance(v,(int,float)): return (ORIGIN+timedelta(seconds=v)).isoformat()
        return v
    return normalize(dates(result))


def permute_graph(encoded):
    if encoded['family']=='tabular': raise ValueError('graph required')
    result=deepcopy(encoded)
    n=len(result['mask']); order=list(reversed(range(n))); inverse={old:new for new,old in enumerate(order)}
    for field in ('mask','tables','states','values'): result[field]=[result[field][i] for i in order]
    result['reference_edges']=[[inverse[i],j,inverse[t]] for i,j,t in reversed(result['reference_edges'])]
    if 'sequences' in result: result['sequences']=[[inverse[i] for i in chain] for chain in result['sequences']]
    return result
