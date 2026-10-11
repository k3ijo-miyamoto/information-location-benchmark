"""Generator1.3 deterministic finite audit worlds. No sampled benchmark or learner."""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from fractions import Fraction as F
from itertools import product

from ..v07.models import Dataset
from ..v07.validation import validate_dataset
from ..v07.storage import TABLES

VERSION = '1.3.0'
DISTRIBUTION = 'information-structure-joint-1'
DESIGN = 'generator13-information-design-0.2'
ORIGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)
K = {'O1': F(1,128), 'O2': F(1,64)}
REGIMES = ('IS0','IS1','IS2','IS3')
PREDS = {'merge_c': {'A': (), 'B': (), 'C': ('A','B'), 'D': ('C',)},
         'join_d': {'A': (), 'B': (), 'C': ('A',), 'D': ('C','B')}}
CADENCES = ((20,40,80),(30,60,90))


@dataclass(frozen=True)
class Coordinates:
    x: int = -1
    h: int = -1
    topology: str = 'merge_c'
    d: int = 30
    c: int = 90
    lead: int = 0

    def __post_init__(self):
        if self.x not in (-1,1) or self.h not in (-1,1) or self.topology not in PREDS or self.d not in (30,45) or self.c not in (90,120,150,180) or self.lead not in (0,1):
            raise ValueError('outside frozen public support')


@dataclass(frozen=True)
class RuntimeConfig:
    regime: str
    opportunity: str
    coordinates: Coordinates

    def __post_init__(self):
        if self.regime not in REGIMES or self.opportunity not in K or not isinstance(self.coordinates, Coordinates):
            raise ValueError('explicit frozen regime/opportunity/coordinates required')


def realize(config, ra, rb):
    """Explicit support realization, never random sampling; config stays private."""
    probabilities(config.regime, config.opportunity, x=config.coordinates.x, h=config.coordinates.h)
    return build_world(config.coordinates, ra, rb)


def support():
    return tuple(Coordinates(*v) for v in product((-1,1),(-1,1),PREDS,(30,45),(90,120,150,180),(0,1)))


def probabilities(regime, opportunity, *, x, h):
    """Topology is deliberately not an argument to the material process."""
    if regime not in REGIMES or opportunity not in K or x not in (-1,1) or h not in (-1,1):
        raise ValueError('unknown regime/opportunity/process value')
    k = K[opportunity]
    if regime == 'IS0': return F(1,2)+9*k*x, F(0)
    if regime == 'IS1': return F(1,2)-9*k, 18*k/(F(1,2)+9*k)
    if regime == 'IS2': return F(1,2)+9*k*h, F(0)
    return F(1,2)-7*k, (14*k+8*k*h)/(F(1,2)+7*k)


def baseline(q):
    return {'A': (150,150+q.d), 'B': (150,150+q.d),
            'C': (150+q.d,150+q.d+q.c), 'D': (150+q.d+q.c,210+q.d+q.c)}


def execute(q, ra, rb):
    if ra not in (0,1) or rb not in (0,1): raise ValueError('binary realization required')
    plan = baseline(q)
    material = {'A':150+180*ra, 'B':150+180*rb}
    actual = {}
    for activity, parents in PREDS[q.topology].items():
        constraints = [plan[activity][0]] + [actual[p][1] for p in parents]
        if activity in material: constraints.append(material[activity])
        start = max(constraints)
        actual[activity] = (start, start+plan[activity][1]-plan[activity][0])
    return actual


def target(q, actual):
    return int(actual['D'][1]-baseline(q)['D'][1] >= 120)


def split(q):
    j = (90,120,150,180).index(q.c)
    a = (30,45).index(q.d)
    if (a,q.lead) == (j//2,j%2): return 'validation'
    if (a,q.lead) == (1-j//2,1-j%2): return 'test'
    return 'train'


def build_world(q, ra, rb, *, origin=ORIGIN):
    """Canonical serialization of one deterministic support atom; no regime in P."""
    if origin.tzinfo is None: raise ValueError('aware origin required')
    plan = baseline(q)
    ts = lambda minute: (origin+timedelta(minutes=minute)).isoformat()
    provenance = {'basis':'hypothesis','assumption_ids':['H-V13']}
    def rec(minute=0, **kwargs):
        return dict(schema_version='0.7',created_at=ts(minute),source_type='synthetic',provenance=provenance,**kwargs)
    def temporal(minute=0, **kwargs):
        return rec(minute,valid_from=ts(minute),known_at=ts(minute),**kwargs)
    raw = {table: [] for table in TABLES}
    raw['metadata'] = rec(profile='schedule-material-v0.7',scenario='schedule_multi_material_branching',generator_version='0.7.0',evidence_register={},assumption_register={'H-V13':{'description':'Methodological finite information-structure process; not empirical calibration','version':VERSION}})
    raw['entities'].append(temporal(entity_id='P',project_id='P',entity_type='project',attributes={'name':'Audit project'}))
    for i,activity in enumerate('ABCD',1):
        raw['entities'].append(temporal(i,entity_id=activity,project_id='P',entity_type='activity',attributes={'name':'Work '+activity,'activity_type':'installation'}))
    for material in ('M1','M2'):
        raw['entities'].append(temporal(entity_id=material,project_id='P',entity_type='material',attributes={'name':'Required material','material_category':'cable_support'}))
    def relation(source,dest,kind):
        raw['relationships'].append(temporal(5,relationship_id='REL'+str(len(raw['relationships'])),source_id=source,target_id=dest,relation_type=kind))
    for entity in ('A','B','C','D','M1','M2'): relation(entity,'P','PART_OF')
    for child,parents in PREDS[q.topology].items():
        for parent in parents: relation(parent,child,'PRECEDES')
    relation('A','M1','REQUIRES_MATERIAL'); relation('B','M2','REQUIRES_MATERIAL')
    for version,minute,known,offset in ((1,10,10,0),(2,110,112,30)):
        raw['schedule_revisions'].append(rec(known,schedule_revision_id='PLAN'+str(version),project_id='P',version=version,previous_revision_id=None if version==1 else 'PLAN1',valid_from=ts(minute),known_at=ts(known),assignments=[{'activity_id':a,'planned_start':ts(s+offset),'planned_end':ts(e+offset)} for a,(s,e) in plan.items()]))
    # Event insertion order resolves equal-time material/predecessor constraints.
    def event(ident,kind,subject,minute,known=None,**payload):
        known = minute if known is None else known
        raw['events'].append(rec(known,event_id=ident,event_type=kind,sequence=len(raw['events'])+1,occurred_at=ts(minute),recorded_at=ts(known),project_id='P',subject_id=subject,actor_ids=[],payload=payload))
    event('E0','PROJECT_CREATED','P',0)
    for i,a in enumerate('ABCD',1): event('CREATE'+a,'ACTIVITY_CREATED',a,i)
    def observation(ident,kind,subjects,minute,known,content,sources=()):
        raw['observations'].append(rec(known,observation_id=ident,observation_type=kind,project_id='P',subject_ids=list(subjects),source_event_ids=list(sources),observed_at=ts(minute),recorded_at=ts(known),visibility='ai_visible',content=content))
    w1,w2,latest = CADENCES[q.lead]
    early = ('low','high') if q.h==1 else ('high','low')
    severity = 'high' if q.x==1 else 'low'
    for i in (1,2):
        delivery = 'DEL'+str(i)
        raw['delivery_plans'].append(temporal(10,delivery_plan_id=delivery,project_id='P',material_id='M'+str(i),expected_at=ts(150)))
        for j,(minute,level) in enumerate(zip((w1,w2,latest),(*early,severity)),1):
            observation(f'W{i}-{j}','delivery_warning',['M'+str(i)],minute,minute+2,dict(warning_kind='delivery_delay_risk',severity=level,delivery_plan_id=delivery))
        observation('U'+str(i),'delivery_update',['M'+str(i)],latest+3,latest+5,dict(delivery_plan_id=delivery,status='dispatch_delayed',estimated_arrival_at=ts(210)))
        for j,minute,level,evidence in ((0,w2+3,'medium',[f'W{i}-2']),(1,latest+6,severity,[f'W{i}-3','U'+str(i)])):
            ident=f'R{i}-{j}'
            raw['risk_assessments'].append(rec(minute+2,risk_assessment_id=ident,previous_assessment_id=None if j==0 else f'R{i}-0',project_id='P',target_activity_ids=['D'],assessed_at=ts(minute),known_at=ts(minute+2),risk_kind='schedule_delay',level=level,assessor_kind='synthetic',baseline_revision_id='PLAN1',supporting_observation_ids=evidence))
            event('E'+ident,'RISK_ASSESSED','D',minute,minute+2,risk_assessment_id=ident)
            observation('O'+ident,'schedule_risk',['D'],minute,minute+2,dict(risk_assessment_id=ident,activity_ids=['D'],risk_level=level),['E'+ident])
    event('REV','SCHEDULE_REVISED','P',110,112,schedule_revision_id='PLAN2')
    observation('OREV','schedule_revision',['P','A','B','C','D'],110,112,dict(schedule_revision_id='PLAN2',activity_ids=list('ABCD')),['REV'])
    # Public evidence is complete before any realized execution is constructed.
    actual = execute(q,ra,rb)
    for i,r in ((1,ra),(2,rb)):
        # A future non-arrival fact is optional; explicit arrival is authoritative.
        event('ARR'+str(i),'MATERIAL_AVAILABLE','M'+str(i),150+180*r,delivery_plan_id='DEL'+str(i))
    for a,(start,end) in actual.items():
        event('START'+a,'ACTIVITY_STARTED',a,start)
        event('END'+a,'ACTIVITY_COMPLETED',a,end)
    raw['events'].sort(key=lambda e:(e['occurred_at'],e['sequence']))
    for event_row in raw['events'][1:]:
        raw['event_links'].append(rec((datetime.fromisoformat(event_row['recorded_at'])-origin).total_seconds()/60,event_link_id='L'+event_row['event_id'],project_id='P',source_event_id='E0',target_event_id=event_row['event_id'],relation_type='PRECEDES',basis='synthetic_ground_truth',confidence=1,known_at=event_row['recorded_at'],visibility='evaluation_only'))
    outcome=[]
    for a,(start,end) in actual.items():
        ds,de=start-plan[a][0],end-plan[a][1]
        outcome.append(dict(activity_id=a,baseline_revision_id='PLAN1',revised_revision_id='PLAN2',actual_start_event_id='START'+a,actual_end_event_id='END'+a,start_deviation_minutes=ds,end_deviation_minutes=de,revised_start_deviation_minutes=ds-30,revised_end_deviation_minutes=de-30,realized_delay=ds>0 or de>0))
    raw['ground_truth']=[rec(actual['D'][1]+1,ground_truth_id='GT',project_id='P',evaluation_as_of=ts(120),root_event_id='E0',physical_root_cause={'status':'unknown'},event_link_ids=[r['event_link_id'] for r in raw['event_links']],affected_entity_ids=list('ABCD'),future_event_ids=[e['event_id'] for e in raw['events'] if e['occurred_at']>ts(120)],final_outcome=dict(outcome_type='schedule',activities=outcome,affected_downstream_activity_ids=['C','D']),visibility='evaluation_only')]
    data=Dataset.model_validate(raw)
    validate_dataset(data)
    return data
