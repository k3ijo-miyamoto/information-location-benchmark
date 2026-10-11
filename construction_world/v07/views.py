from pydantic import AwareDatetime, TypeAdapter
from .models import Activity, Actor, Artifact, Issue, Location, Project, Material, DeliveryWarningObservation, DeliveryUpdateObservation
from .validation import content_refs, event_key, validate_dataset


def _at(value):
    return TypeAdapter(AwareDatetime).validate_python(value)


def active(row, at, known=False):
    if row.valid_from > at or (known and row.known_at > at):
        return False
    # A retroactive end is effective in real history, but invisible until known.
    return row.valid_to is None or row.valid_to > at or (known and row.valid_to_known_at > at)


def replay(entities, events):
    states = {e.entity_id: {'status': e.attributes.initial_status} for e in entities if isinstance(e, (Project, Activity, Issue))}
    transitions = {'PROJECT_STARTED': 'active', 'ACTIVITY_STARTED': 'in_progress',
        'ACTIVITY_COMPLETED': 'completed', 'INSPECTION_PERFORMED': 'in_progress',
        'REINSPECTION_PERFORMED': 'in_progress', 'INSPECTION_FAILED': 'completed',
        'INSPECTION_PASSED': 'completed', 'CORRECTION_STARTED': 'in_progress',
        'CORRECTION_COMPLETED': 'completed', 'SCHEDULE_ISSUE_RESOLVED': 'resolved', 'ISSUE_RESOLVED': 'pending_verification',
        'ISSUE_CLOSED': 'closed', 'ISSUE_REOPENED': 'open'}
    for e in entities:
        if isinstance(e, Activity):
            states[e.entity_id]['blocked'] = False
        if isinstance(e, Material):
            states[e.entity_id] = {'availability': 'unknown'}
    open_blocks = {}
    for ev in sorted(events, key=event_key):
        if ev.subject_id in states:
            if ev.event_type in {'MATERIAL_UNAVAILABLE', 'MATERIAL_AVAILABLE'}:
                states[ev.subject_id]['availability'] = 'available' if ev.event_type == 'MATERIAL_AVAILABLE' else 'unavailable'
            if ev.event_type in {'BLOCKING_STARTED', 'BLOCKING_CLEARED'}:
                blocks = open_blocks.setdefault(ev.subject_id, set())
                if ev.event_type == 'BLOCKING_STARTED':
                    blocks.add(ev.payload.blocking_condition_id)
                else:
                    blocks.discard(ev.payload.blocking_condition_id)
                states[ev.subject_id]['blocked'] = bool(blocks)
            if ev.event_type in transitions:
                states[ev.subject_id]['status'] = transitions[ev.event_type]
            if ev.payload.result is not None:
                states[ev.subject_id]['inspection_result'] = ev.payload.result
    return states


def world_state(data, as_of):
    validate_dataset(data)
    at = _at(as_of)
    return replay([e for e in data.entities if active(e, at)],
                  [e for e in data.events if e.occurred_at <= at])


def commercial_state(data, as_of, known=False):
    """Commercial records effective by cutoff; optionally apply knowledge gates.

    Does not derive actual cost, revenue, gross profit or physical Activity state.
    """
    from .commercial import commercial_view
    validate_dataset(data)
    return commercial_view(data, _at(as_of), known=known)


def build_ai_view(data, as_of):
    validate_dataset(data)
    at = _at(as_of)
    entities = [e for e in data.entities if active(e, at, known=True)]
    # Fixed point: references in immutable attributes must also be known/active.
    while True:
        ids = {e.entity_id for e in entities}
        filtered = [e for e in entities if e.project_id in ids and
                    (not isinstance(e, Activity) or all(ident in ids for ident in
                     (e.attributes.inspection_of_id, e.attributes.previous_inspection_id) if ident is not None))]
        if len(filtered) == len(entities):
            break
        entities = filtered
    selections = select_records(data, at, ids)
    public_refs = ids | {getattr(row, key) for table, key in RECORD_IDS.items() for row in selections[table]}
    events = [e for e in data.events if all(ident in public_refs for ident in payload_refs(e.payload))
              and e.occurred_at <= at and e.recorded_at <= at
              and {e.project_id, e.subject_id, *e.actor_ids, *e.payload.issue_ids} <= ids]
    state = replay(entities, events)
    entity_rows = []
    for e in entities:
        a = e.attributes
        if isinstance(e, Project):
            attrs = {'project_category': a.project_category}
        elif isinstance(e, Activity):
            attrs = {key: getattr(a, key) for key in ('activity_type', 'inspection_type',
                     'inspection_of_id', 'inspection_attempt', 'previous_inspection_id') if getattr(a, key) is not None}
        elif isinstance(e, Actor):
            attrs = {'actor_kind': a.actor_kind, 'role': a.role}
        elif isinstance(e, Issue):
            attrs = {'issue_kind': a.issue_kind}
        elif isinstance(e, Material):
            attrs = {'material_category': a.material_category}
        elif isinstance(e, Location):
            attrs = {'location_type': a.location_type}
        else:
            attrs = {'artifact_type': a.artifact_type}
        attrs.update(state.get(e.entity_id, {}))
        entity_rows.append({'entity_id': e.entity_id, 'entity_type': e.entity_type,
                            'project_id': e.project_id, 'attributes': attrs})
    event_ids = {e.event_id for e in events}
    # Typed, bounded profile fields only. Do not dump an entire Observation.
    obs_fields = {
        'inspection_record': ('inspection_id', 'check_item', 'result', 'finding', 'issue_ids'),
        'progress_record': ('activity_id', 'reported_progress'),
        'message': ('issue_ids', 'message_kind', 'statement'),
        'photo': ('activity_id', 'issue_ids', 'depicts', 'artifact_id'),
        'report': ('activity_id', 'report_kind', 'reported_progress'),
        'material_availability': ('material_id', 'delivery_plan_id', 'availability'),
        'schedule_revision': ('schedule_revision_id', 'activity_ids'),
        'schedule_risk': ('risk_assessment_id', 'activity_ids', 'risk_level'),
        'delivery_warning': ('warning_kind', 'severity', 'delivery_plan_id'),
        'delivery_update': ('delivery_plan_id', 'status', 'estimated_arrival_at'),
    }
    from .commercial import commercial_view
    from .projection import project_balances
    balances = {'commercial_balances': project_balances(data, at)} if data.metadata.profile == 'commercial-v0.7' else {}
    return {**balances, **commercial_view(data, at), **project_records(selections), 'as_of': at.isoformat(), 'entities': entity_rows,
        'relationships': [{'relationship_id': r.relationship_id, 'source_id': r.source_id,
                           'target_id': r.target_id, 'relation_type': r.relation_type}
                          for r in data.relationships if active(r, at, known=True) and {r.source_id, r.target_id} <= ids],
        'events': [{'event_id': e.event_id, 'event_type': e.event_type, 'occurred_at': e.occurred_at.isoformat(),
                    'recorded_at': e.recorded_at.isoformat(), 'project_id': e.project_id,
                    'subject_id': e.subject_id, 'actor_ids': list(e.actor_ids),
                    'payload': public_payload(e.payload)} for e in events],
        'event_links': [{'event_link_id': link.event_link_id, 'source_event_id': link.source_event_id,
                         'target_event_id': link.target_event_id, 'relation_type': link.relation_type,
                         'basis': link.basis, 'confidence': link.confidence}
                        for link in data.event_links if link.visibility == 'ai_visible'
                        and link.known_at <= at and {link.source_event_id, link.target_event_id} <= event_ids],
        'observations': [{'observation_id': o.observation_id, 'observation_type': o.observation_type,
                          'observed_at': o.observed_at.isoformat(), 'recorded_at': o.recorded_at.isoformat(),
                          'project_id': o.project_id, 'subject_ids': list(o.subject_ids),
                          'content': {key: (list(getattr(o.content, key)) if isinstance(getattr(o.content, key), list)
                                           else o.content.model_dump(mode='json')[key])
                                      for key in obs_fields[o.observation_type] if getattr(o.content, key) is not None}}
                         for o in data.observations if o.visibility == 'ai_visible'
                         and o.observed_at <= at and o.recorded_at <= at
                         and (not isinstance(o, DeliveryUpdateObservation) or set(o.source_event_ids) <= event_ids)
                         and {o.project_id, *o.subject_ids, *content_refs(o)} <= ids
                         and all(ident in public_refs for ident in auxiliary_refs(o.content))]}


RECORD_IDS = {'schedule_revisions': 'schedule_revision_id', 'delivery_plans': 'delivery_plan_id',
              'blocking_conditions': 'blocking_condition_id', 'risk_assessments': 'risk_assessment_id'}
AUX_FIELDS = ('delivery_plan_id', 'blocking_condition_id', 'schedule_revision_id', 'risk_assessment_id')

def auxiliary_refs(obj):
    return [getattr(obj, key) for key in AUX_FIELDS if getattr(obj, key, None) is not None]

def payload_refs(payload):
    return auxiliary_refs(payload)

def public_payload(payload):
    return {'result': payload.result, 'issue_ids': list(payload.issue_ids),
            **{key: getattr(payload, key) for key in AUX_FIELDS if getattr(payload, key) is not None}}

def select_records(data, at, ids):
    plans = [p for p in data.schedule_revisions if p.valid_from <= at and p.known_at <= at
             and {p.project_id, *(w.activity_id for w in p.assignments)} <= ids]
    # A revision is not public until the full referenced chain is public.
    while True:
        plan_ids = {p.schedule_revision_id for p in plans}
        new = [p for p in plans if p.previous_revision_id is None or p.previous_revision_id in plan_ids]
        if len(new) == len(plans):
            break
        plans = new
    deliveries = [p for p in data.delivery_plans if p.valid_from <= at and p.known_at <= at
                  and {p.material_id, p.project_id} <= ids]
    delivery_ids = {p.delivery_plan_id for p in deliveries}
    blocks = [b for b in data.blocking_conditions if b.valid_from <= at and b.known_at <= at
              and {b.activity_id, b.material_id, b.project_id} <= ids and b.delivery_plan_id in delivery_ids]
    known_event_ids = {e.event_id for e in data.events if e.occurred_at <= at and e.recorded_at <= at}
    warning_ids = {o.observation_id for o in data.observations if isinstance(o, (DeliveryWarningObservation, DeliveryUpdateObservation))
                   and o.visibility == 'ai_visible' and o.observed_at <= at and o.recorded_at <= at
                   and {o.project_id, *o.subject_ids} <= ids and o.content.delivery_plan_id in delivery_ids
                   and set(o.source_event_ids) <= known_event_ids}
    risks = [r for r in data.risk_assessments if r.assessed_at <= at and r.known_at <= at
             and {r.project_id, *r.target_activity_ids} <= ids and r.baseline_revision_id in plan_ids
             and set(r.supporting_event_ids) <= known_event_ids
             and set(r.supporting_observation_ids) <= warning_ids]
    # Only explicit, closure-complete chains; an unknown successor never removes history.
    while True:
        risk_ids = {r.risk_assessment_id for r in risks}
        available = ids | plan_ids | delivery_ids | {b.blocking_condition_id for b in blocks} | risk_ids
        visible_events = {e.event_id for e in data.events if e.event_id in known_event_ids
                          and {e.project_id, e.subject_id, *e.actor_ids, *e.payload.issue_ids} <= ids
                          and set(payload_refs(e.payload)) <= available}
        hidden_updates = {o.observation_id for o in data.observations if isinstance(o, DeliveryUpdateObservation)
                          and not set(o.source_event_ids) <= visible_events}
        closed = [r for r in risks if r.previous_assessment_id is None or r.previous_assessment_id in risk_ids]
        closed = [r for r in closed if not set(r.supporting_observation_ids) & hidden_updates]
        if len(closed) == len(risks):
            break
        risks = closed
    return dict(schedule_revisions=plans, delivery_plans=deliveries, blocking_conditions=blocks, risk_assessments=risks)

def project_records(selected):
    # Field allowlists deliberately omit provenance and supporting internal links.
    fields = {
        'schedule_revisions': ('schedule_revision_id', 'project_id', 'version', 'previous_revision_id', 'valid_from', 'known_at', 'assignments'),
        'delivery_plans': ('delivery_plan_id', 'project_id', 'material_id', 'expected_at', 'valid_from', 'known_at'),
        'blocking_conditions': ('blocking_condition_id', 'project_id', 'activity_id', 'reason', 'material_id', 'delivery_plan_id', 'valid_from', 'known_at'),
        'risk_assessments': ('risk_assessment_id', 'project_id', 'target_activity_ids', 'assessed_at', 'known_at', 'risk_kind', 'level', 'assessor_kind', 'baseline_revision_id', 'supporting_observation_ids', 'previous_assessment_id'),
    }
    result = {table: [row.model_dump(mode='json', include=set(fields[table])) for row in rows]
              for table, rows in selected.items()}
    for row in result['risk_assessments']:
        if row['previous_assessment_id'] is None:
            row.pop('previous_assessment_id')  # Preserve migrated legacy public facts exactly.
        if not row['supporting_observation_ids']:
            row.pop('supporting_observation_ids')
    return result


def schedule_comparison(data, as_of, known=True):
    """Actual vs baseline/latest-known plan; unobserved actuals produce null deltas."""
    validate_dataset(data)
    at = _at(as_of)
    plans = [p for p in data.schedule_revisions if p.valid_from <= at and (not known or p.known_at <= at)]
    result = {}
    for project in {p.project_id for p in plans}:
        revisions = sorted([p for p in plans if p.project_id == project], key=lambda p: p.version)
        baseline, latest = revisions[0], revisions[-1]
        for window in baseline.assignments:
            recent = next(w for w in latest.assignments if w.activity_id == window.activity_id)
            actual = [e for e in data.events if e.subject_id == window.activity_id and e.occurred_at <= at
                      and (not known or e.recorded_at <= at)]
            row = {'baseline_revision_id': baseline.schedule_revision_id, 'latest_revision_id': latest.schedule_revision_id}
            for label, kind, planned, revised in [('start', 'ACTIVITY_STARTED', window.planned_start, recent.planned_start),
                                                   ('end', 'ACTIVITY_COMPLETED', window.planned_end, recent.planned_end)]:
                ev = next((e for e in actual if e.event_type == kind), None)
                row['actual_' + label] = ev.occurred_at.isoformat() if ev else None
                row[label + '_deviation_minutes'] = (ev.occurred_at-planned).total_seconds()/60 if ev else None
                row['revised_' + label + '_deviation_minutes'] = (ev.occurred_at-revised).total_seconds()/60 if ev else None
            result[window.activity_id] = row
    return result
