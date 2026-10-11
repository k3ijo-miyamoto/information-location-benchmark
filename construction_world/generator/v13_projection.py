"""Closed normalized Canonical public authority and evaluator-only projections."""
from copy import deepcopy
from datetime import datetime
import json

ID_FIELDS = {'entities':'entity_id','relationships':'relationship_id','events':'event_id',
             'event_links':'event_link_id','observations':'observation_id',
             'schedule_revisions':'schedule_revision_id','delivery_plans':'delivery_plan_id',
             'risk_assessments':'risk_assessment_id','blocking_conditions':'blocking_condition_id'}
EMPTY = ('estimates','execution_budgets','orders','commercial_changes','acceptances','invoices','invoice_assessments','payments','commercial_observations')
TABLES = tuple(ID_FIELDS)+EMPTY
CLOCKS = {'as_of','occurred_at','recorded_at','observed_at','assessed_at','known_at','valid_from','planned_start','planned_end','expected_at','estimated_arrival_at'}
FIELDS = {
 'entities': {'entity_id','entity_type','project_id','attributes'},
 'relationships': {'relationship_id','source_id','target_id','relation_type'},
 'events': {'event_id','event_type','occurred_at','recorded_at','project_id','subject_id','actor_ids','payload'},
 'event_links': {'event_link_id','source_event_id','target_event_id','relation_type','basis','confidence'},
 'observations': {'observation_id','observation_type','observed_at','recorded_at','project_id','subject_ids','content'},
 'schedule_revisions': {'schedule_revision_id','project_id','version','previous_revision_id','valid_from','known_at','assignments'},
 'delivery_plans': {'delivery_plan_id','project_id','material_id','expected_at','valid_from','known_at'},
 'risk_assessments': {'risk_assessment_id','project_id','target_activity_ids','assessed_at','known_at','risk_kind','level','assessor_kind','baseline_revision_id','supporting_observation_ids','previous_assessment_id'},
 'blocking_conditions': set(),
}

def key(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'))


def normalize(public):
    if set(public) != set(TABLES)|{'as_of'}: raise ValueError('non-public/unknown top-level fields')
    p=deepcopy(public)
    if any(p[t] for t in EMPTY) or p['blocking_conditions'] or p['event_links']:
        raise ValueError('outside frozen public profile')
    for table in ID_FIELDS:
        for row in p[table]:
            if set(row)-FIELDS[table]: raise ValueError('unknown public record field')
    origin_events=[e for e in p['events'] if e['event_type']=='PROJECT_CREATED']
    if len(origin_events)!=1: raise ValueError('unique public origin required')
    origin=datetime.fromisoformat(origin_events[0]['occurred_at'])
    def clocks(value,field=None):
        if isinstance(value,dict): return {k:clocks(v,k) for k,v in value.items()}
        if isinstance(value,list): return [clocks(v,field) for v in value]
        if field in CLOCKS and isinstance(value,str):
            seconds=(datetime.fromisoformat(value)-origin).total_seconds()
            if not seconds.is_integer(): raise ValueError('subsecond clock outside support')
            return int(seconds)
        return value
    p=clocks(p)
    if p['as_of']!=7200: raise ValueError('wrong cutoff')
    for table,field in [('events','recorded_at'),('observations','recorded_at'),('risk_assessments','known_at'),('schedule_revisions','known_at'),('delivery_plans','known_at')]:
        if any(r[field]>7200 for r in p[table]): raise ValueError('future public fact')
    if any(e['event_type'] not in {'PROJECT_CREATED','ACTIVITY_CREATED','RISK_ASSESSED','SCHEDULE_REVISED'} for e in p['events']):
        raise ValueError('actual/future event in public input')
    mapping={}
    def register(table,sortkey):
        rows=sorted(p[table],key=sortkey)
        keys=[sortkey(r) for r in rows]
        if len(keys)!=len(set(keys)): raise ValueError('ambiguous public slot key')
        for i,row in enumerate(rows):
            ident=row[ID_FIELDS[table]]
            if ident in mapping: raise ValueError('duplicate ID')
            mapping[ident]=f'{table}:{i}'
        p[table]=rows
    births={e['subject_id']:e['occurred_at'] for e in p['events'] if e['event_type']=='ACTIVITY_CREATED'}
    requirements={r['target_id']:r['source_id'] for r in p['relationships'] if r['relation_type']=='REQUIRES_MATERIAL'}
    def entitykey(r):
        kind=r['entity_type']
        if kind=='activity': return (1,births[r['entity_id']])
        if kind=='material': return (2,births[requirements[r['entity_id']]])
        if kind=='project': return (0,0)
        raise ValueError('unsupported entity')
    register('entities',entitykey)
    register('schedule_revisions',lambda r:r['version'])
    register('delivery_plans',lambda r:mapping[r['material_id']])
    observations=p['observations']
    process=[r for r in observations if r['observation_type'] in ('delivery_warning','delivery_update')]
    process.sort(key=lambda r:(mapping[r['content']['delivery_plan_id']],r['observed_at'],r['recorded_at'],r['observation_type']))
    for i,r in enumerate(process): mapping[r['observation_id']]=f'observations:{i}'
    register('risk_assessments',lambda r:(tuple(sorted(mapping[x] for x in r['supporting_observation_ids'])),r['assessed_at'],r['known_at']))
    rest=[r for r in observations if r not in process]
    rest.sort(key=lambda r:(r['observation_type'],mapping[r['content'].get('risk_assessment_id',r['content'].get('schedule_revision_id'))],r['observed_at']))
    for i,r in enumerate(rest,len(process)): mapping[r['observation_id']]=f'observations:{i}'
    if len(set(r['observation_id'] for r in observations))!=len(observations): raise ValueError('duplicate observation')
    p['observations']=process+rest
    register('events',lambda r:(r['occurred_at'],mapping[r['subject_id']],r['event_type'],r['recorded_at'],tuple(sorted((k,mapping.get(v,v)) for k,v in r['payload'].items() if not isinstance(v,list)))))
    register('relationships',lambda r:(r['relation_type'],mapping[r['source_id']],mapping[r['target_id']]))
    def replace(v,field=None):
        if isinstance(v,dict): return {k:replace(x,k) for k,x in v.items()}
        if isinstance(v,list):
            values=[replace(x,field) for x in v]
            # All list fields here are sets or assignment maps, not chronology.
            return sorted(values,key=key)
        if isinstance(v,str):
            if field and (field.endswith('_id') or field.endswith('_ids')):
                if v not in mapping: raise ValueError('unclosed public reference')
                return mapping[v]
            return v
        return v
    result={t:[replace(r) for r in p[t]] for t in TABLES}
    result['as_of']=p['as_of']
    return result


def projections(p):
    warnings=[o for o in p['observations'] if o['observation_type']=='delivery_warning']
    latest=[]
    for plan in p['delivery_plans']:
        group=[w for w in warnings if w['content']['delivery_plan_id']==plan['delivery_plan_id']]
        latest.append(max(group,key=lambda w:w['observed_at']))
    latest_risks=[r for r in p['risk_assessments'] if r.get('previous_assessment_id')]
    current=dict(warnings=latest,updates=[o for o in p['observations'] if o['observation_type']=='delivery_update'],risk=latest_risks,plans=p['schedule_revisions'],deliveries=p['delivery_plans'])
    topology=sorted((r['source_id'],r['target_id'],r['relation_type']) for r in p['relationships'] if r['relation_type'] in ('PRECEDES','DEPENDS_ON'))
    history=sorted(warnings+[o for o in p['observations'] if o['observation_type']=='delivery_update'],key=lambda o:(o['observed_at'],o['observation_id']))
    history=dict(records=history,risks=p['risk_assessments'])
    return {'empty':key(None),'Xc':key(current),'XcT':key([current,topology]),'XcH':key([current,history]),'full':key([current,topology,history]),'P':key(p)}


def sequences(p):
    """Seven public record IDs per stream, no trajectory/latent summary."""
    result=[]
    for plan in p['delivery_plans']:
        observations=sorted([o for o in p['observations'] if o['observation_type'] in ('delivery_warning','delivery_update') and o['content']['delivery_plan_id']==plan['delivery_plan_id']],key=lambda o:o['observed_at'])
        ids={o['observation_id'] for o in observations}
        risks=sorted([r for r in p['risk_assessments'] if set(r['supporting_observation_ids'])<=ids],key=lambda r:r['assessed_at'])
        if len(observations)!=4 or len(risks)!=2: raise ValueError('wrong chain cardinality')
        result.append([plan['delivery_plan_id'],observations[0]['observation_id'],observations[1]['observation_id'],risks[0]['risk_assessment_id'],observations[2]['observation_id'],observations[3]['observation_id'],risks[1]['risk_assessment_id']])
    return result
