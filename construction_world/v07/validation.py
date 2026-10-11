"""Cross-record contracts; no fixed IDs or single-finding count assumptions."""
from collections import Counter
from .models import (Dataset, Activity, Actor, Artifact, Issue, Project,
                     InspectionObservation, ProgressObservation, MessageObservation,
                     PhotoObservation, ReportObservation, Material, ScheduleOutcome, InspectionOutcome,
                     MaterialObservation, ScheduleObservation, RiskObservation)

# Endpoint contracts shared with documentation and tests. Activity subtypes below.
ENDPOINTS = {
    'PART_OF': ({'activity', 'actor', 'issue', 'location', 'artifact', 'material'}, {'project'}),
    'PRECEDES': ({'activity'}, {'activity'}),
    'DEPENDS_ON': ({'activity'}, {'activity'}),
    'ASSIGNED_TO': ({'activity'}, {'actor'}),
    'LOCATED_AT': ({'activity', 'issue', 'actor', 'artifact'}, {'location'}),
    'REFERENCES': ({'activity', 'issue', 'artifact'}, {'artifact'}),
    'OBSERVES': ({'activity'}, {'activity', 'issue', 'location'}),
    'DETECTS': ({'activity'}, {'issue'}),
    'CORRECTS': ({'activity'}, {'issue'}),
    'VERIFIES': ({'activity'}, {'issue'}),
    'APPROVES': ({'actor'}, {'activity', 'artifact'}),
    'SUPERSEDES': ({'artifact'}, {'artifact'}),
    'REQUIRES_MATERIAL': ({'activity'}, {'material'}),
}

class ValidationError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__('v0.7 validation failed:\n' + '\n'.join('- ' + e for e in errors))


def event_key(e):
    return e.occurred_at, e.sequence


def content_refs(o):
    c = o.content
    refs = set(getattr(c, 'issue_ids', []))
    refs.update(getattr(c, 'activity_ids', []))
    for name in ('inspection_id', 'activity_id', 'artifact_id', 'material_id'):
        if getattr(c, name, None) is not None:
            refs.add(getattr(c, name))
    return refs


def validate_dataset(dataset: Dataset) -> None:
    d = Dataset.model_validate(dataset.model_dump())
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    groups = [('entities', 'entity_id'), ('relationships', 'relationship_id'), ('events', 'event_id'),
              ('event_links', 'event_link_id'), ('observations', 'observation_id'), ('ground_truth', 'ground_truth_id'),
              ('schedule_revisions', 'schedule_revision_id'), ('delivery_plans', 'delivery_plan_id'),
              ('blocking_conditions', 'blocking_condition_id'), ('risk_assessments', 'risk_assessment_id')]
    from .commercial import COMMERCIAL_IDS, validate_commercial
    groups.extend(COMMERCIAL_IDS.items())
    all_ids = [getattr(row, field) for table, field in groups for row in getattr(d, table)]
    for ident, count in Counter(all_ids).items():
        check(count == 1, f'duplicate ID {ident}')
    entities = {e.entity_id: e for e in d.entities}
    events = {e.event_id: e for e in d.events}
    links = {e.event_link_id: e for e in d.event_links}
    def ref(ident, context, project=None, when=None, cls=None):
        obj = entities.get(ident)
        check(obj is not None, f'{context}: missing entity {ident}')
        if obj:
            if project is not None:
                check(obj.project_id == project, f'{context}: cross-project reference {ident}')
            if cls is not None:
                check(isinstance(obj, cls), f'{context}: wrong entity type {ident}')
            if when is not None:
                check(obj.valid_from <= when and (obj.valid_to is None or when < obj.valid_to),
                      f'{context}: entity {ident} not valid at effective time')
        return obj
    def event_ref(ident, context, project):
        obj = events.get(ident)
        check(obj is not None, f'{context}: missing event {ident}')
        if obj:
            check(obj.project_id == project, f'{context}: cross-project event')
        return obj
    def relation(source, target, kind, when=None):
        return any(r.source_id == source and r.target_id == target and r.relation_type == kind
                   and (when is None or r.valid_from <= when and (r.valid_to is None or when < r.valid_to))
                   for r in d.relationships)
    def prior(ev, kinds, subject=None, issue=None):
        return any(e.event_type in kinds and e.project_id == ev.project_id and event_key(e) < event_key(ev)
                   and (subject is None or e.subject_id == subject)
                   and (issue is None or issue in e.payload.issue_ids or e.subject_id == issue) for e in d.events)
    for e in d.entities:
        ref(e.project_id, e.entity_id, cls=Project)
        if isinstance(e, Project):
            check(e.project_id == e.entity_id, f'{e.entity_id}: project must self-reference')
        if isinstance(e, Activity) and e.attributes.activity_type == 'inspection':
            a = e.attributes
            ref(a.inspection_of_id, e.entity_id, e.project_id, e.valid_from, Activity)
            check(a.inspection_of_id != e.entity_id, f'{e.entity_id}: inspection cannot inspect itself')
            if a.previous_inspection_id:
                previous = ref(a.previous_inspection_id, e.entity_id, e.project_id, cls=Activity)
                if isinstance(previous, Activity):
                    pa = previous.attributes
                    check(pa.activity_type == 'inspection' and pa.inspection_of_id == a.inspection_of_id
                          and pa.inspection_attempt == a.inspection_attempt - 1
                          and previous.valid_from < e.valid_from,
                          f'{e.entity_id}: invalid previous inspection/attempt')
    for r in d.relationships:
        a, b = ref(r.source_id, r.relationship_id, when=r.valid_from), ref(r.target_id, r.relationship_id, when=r.valid_from)
        check(r.source_id != r.target_id, f'{r.relationship_id}: self relationship')
        if a and b:
            check(a.project_id == b.project_id, f'{r.relationship_id}: cross-project relationship')
            source_types, target_types = ENDPOINTS[r.relation_type]
            check(a.entity_type in source_types and b.entity_type in target_types,
                  f'{r.relationship_id}: invalid endpoints for {r.relation_type}')
            required = {'DETECTS': 'inspection', 'VERIFIES': 'inspection', 'CORRECTS': 'correction'}.get(r.relation_type)
            if required:
                check(isinstance(a, Activity) and a.attributes.activity_type == required,
                      f'{r.relationship_id}: {r.relation_type} requires {required} source')
    check(d.events == sorted(d.events, key=event_key), 'events not sorted by occurred_at/sequence')
    check(len({e.sequence for e in d.events}) == len(d.events), 'duplicate event sequence')
    # Actual replay validation is per entity, not per recording order.
    statuses = {e.entity_id: e.attributes.initial_status for e in d.entities if isinstance(e, (Project, Activity, Issue))}
    transitions = {
        'PROJECT_STARTED': ({'planned'}, 'active'),
        'ACTIVITY_STARTED': ({'planned'}, 'in_progress'),
        'ACTIVITY_COMPLETED': ({'in_progress'}, 'completed'),
        'INSPECTION_PERFORMED': ({'planned'}, 'in_progress'),
        'REINSPECTION_PERFORMED': ({'planned'}, 'in_progress'),
        'INSPECTION_FAILED': ({'in_progress'}, 'completed'),
        'INSPECTION_PASSED': ({'in_progress'}, 'completed'),
        'CORRECTION_STARTED': ({'planned'}, 'in_progress'),
        'CORRECTION_COMPLETED': ({'in_progress'}, 'completed'),
        'ISSUE_RESOLVED': ({'open'}, 'pending_verification'),
        'ISSUE_CLOSED': ({'pending_verification'}, 'closed'),
        'ISSUE_REOPENED': ({'pending_verification', 'closed'}, 'open'),
        'SCHEDULE_ISSUE_RESOLVED': ({'open'}, 'resolved'),
    }
    for ev in sorted(d.events, key=event_key):
        ctx = ev.event_id
        ref(ev.project_id, ctx, cls=Project)
        subject = ref(ev.subject_id, ctx, ev.project_id, ev.occurred_at)
        expected_cls = Project if ev.event_type.startswith('PROJECT_') else Issue if ev.event_type.startswith('ISSUE_') else Activity
        if ev.event_type.startswith('MATERIAL_'):
            expected_cls = Material
        elif ev.event_type == 'SCHEDULE_REVISED':
            expected_cls = Project
        elif ev.event_type == 'SCHEDULE_ISSUE_RESOLVED':
            expected_cls = Issue
        check(isinstance(subject, expected_cls), f'{ctx}: invalid subject type for {ev.event_type}')
        for ident in ev.actor_ids:
            ref(ident, ctx, ev.project_id, ev.occurred_at, Actor)
        for ident in ev.payload.issue_ids:
            ref(ident, ctx, ev.project_id, ev.occurred_at, Issue)
        check(len(ev.payload.issue_ids) == len(set(ev.payload.issue_ids)), f'{ctx}: duplicate issue reference')
        expected_result = {'INSPECTION_FAILED': 'fail', 'INSPECTION_PASSED': 'pass'}.get(ev.event_type)
        check(ev.payload.result == expected_result, f'{ctx}: invalid result payload')
        if isinstance(subject, Activity):
            if ev.event_type != 'ACTIVITY_CREATED':
                check(prior(ev, {'ACTIVITY_CREATED'}, subject=ev.subject_id), f'{ctx}: activity event before creation')
            required_type = ('inspection' if ev.event_type in {'INSPECTION_PERFORMED', 'REINSPECTION_PERFORMED',
                'INSPECTION_FAILED', 'INSPECTION_PASSED', 'CORRECTION_REQUESTED'} else
                'correction' if ev.event_type in {'CORRECTION_STARTED', 'CORRECTION_COMPLETED'} else None)
            if required_type:
                check(subject.attributes.activity_type == required_type, f'{ctx}: wrong Activity subtype')
            if ev.event_type == 'REINSPECTION_PERFORMED':
                check(subject.attributes.previous_inspection_id is not None, f'{ctx}: reinspection needs previous inspection')
            if ev.event_type == 'INSPECTION_PERFORMED':
                check(subject.attributes.inspection_attempt == 1, f'{ctx}: initial inspection requires attempt 1')
        if ev.event_type in transitions:
            allowed, new = transitions[ev.event_type]
            check(statuses.get(ev.subject_id) in allowed, f'{ctx}: illegal state transition {statuses.get(ev.subject_id)} -> {new}')
            statuses[ev.subject_id] = new
        if ev.event_type in {'INSPECTION_FAILED', 'CORRECTION_REQUESTED', 'CORRECTION_STARTED',
                              'CORRECTION_COMPLETED', 'REINSPECTION_PERFORMED'}:
            check(bool(ev.payload.issue_ids), f'{ctx}: requires issue_ids')
        for ident in ev.payload.issue_ids:
            if ev.event_type == 'INSPECTION_FAILED':
                check(relation(ev.subject_id, ident, 'DETECTS', ev.occurred_at)
                      or relation(ev.subject_id, ident, 'VERIFIES', ev.occurred_at), f'{ctx}: missing DETECTS/VERIFIES')
            if ev.event_type == 'CORRECTION_REQUESTED':
                check(prior(ev, {'INSPECTION_FAILED'}, subject=ev.subject_id, issue=ident), f'{ctx}: request before failure')
            if ev.event_type in {'CORRECTION_STARTED', 'CORRECTION_COMPLETED'}:
                check(relation(ev.subject_id, ident, 'CORRECTS', ev.occurred_at), f'{ctx}: missing CORRECTS')
                check(prior(ev, {'CORRECTION_REQUESTED'}, issue=ident), f'{ctx}: correction before request')
            if ev.event_type in {'REINSPECTION_PERFORMED', 'INSPECTION_PASSED'}:
                check(relation(ev.subject_id, ident, 'VERIFIES', ev.occurred_at), f'{ctx}: missing VERIFIES')
                check(prior(ev, {'CORRECTION_COMPLETED'}, issue=ident), f'{ctx}: reinspection/pass before correction completion')
        if ev.event_type == 'ISSUE_RESOLVED':
            check(prior(ev, {'CORRECTION_COMPLETED'}, issue=ev.subject_id), f'{ctx}: resolution requires correction completion')
        if ev.event_type == 'ISSUE_CLOSED':
            check(prior(ev, {'INSPECTION_PASSED'}, issue=ev.subject_id), f'{ctx}: closure requires verification pass')
        if ev.event_type == 'ISSUE_REOPENED':
            check(prior(ev, {'INSPECTION_FAILED'}, issue=ev.subject_id), f'{ctx}: reopening requires failed inspection')
    for ent in d.entities:
        birth_type = 'PROJECT_CREATED' if isinstance(ent, Project) else 'ACTIVITY_CREATED' if isinstance(ent, Activity) else 'ISSUE_RAISED' if isinstance(ent, Issue) else None
        if birth_type:
            births = [e for e in d.events if e.subject_id == ent.entity_id and e.event_type == birth_type]
            check(len(births) == 1 and births[0].occurred_at == ent.valid_from, f'{ent.entity_id}: requires unique birth event at valid_from')
    for r in d.relationships:
        if r.relation_type in {'PRECEDES', 'DEPENDS_ON'}:
            first, second = (r.source_id, r.target_id) if r.relation_type == 'PRECEDES' else (r.target_id, r.source_id)
            for start in [e for e in d.events if e.subject_id == second and e.event_type in {'ACTIVITY_STARTED', 'CORRECTION_STARTED', 'INSPECTION_PERFORMED', 'REINSPECTION_PERFORMED'}]:
                check(prior(start, {'ACTIVITY_COMPLETED', 'CORRECTION_COMPLETED', 'INSPECTION_FAILED', 'INSPECTION_PASSED'}, subject=first),
                      f'{r.relationship_id}: predecessor not completed before dependent starts')
    def dag(edges, label):
        nodes = {n for edge in edges for n in edge}
        pending = {n: {a for a, b in edges if b == n} for n in nodes}
        while pending:
            ready = {n for n, parents in pending.items() if not parents}
            if not ready:
                errors.append(f'{label}: cycle')
                return
            pending = {n: parents - ready for n, parents in pending.items() if n not in ready}
    order_edges = []
    for link in d.event_links:
        a = event_ref(link.source_event_id, link.event_link_id, link.project_id)
        b = event_ref(link.target_event_id, link.event_link_id, link.project_id)
        check(link.source_event_id != link.target_event_id, f'{link.event_link_id}: self-link')
        if a and b:
            check(link.known_at >= max(a.occurred_at, b.occurred_at), f'{link.event_link_id}: link known before endpoints occur')
            if link.relation_type != 'ASSOCIATED_WITH':
                check(a.occurred_at <= b.occurred_at, f'{link.event_link_id}: cause/precedence after effect')
                order_edges.append((a.event_id, b.event_id))
    dag(order_edges, 'causal/precedence DAG')
    dag([(r.source_id, r.target_id) if r.relation_type == 'PRECEDES' else (r.target_id, r.source_id)
         for r in d.relationships if r.relation_type in {'PRECEDES', 'DEPENDS_ON'}], 'Activity dependency DAG')
    for obs in d.observations:
        ctx = obs.observation_id
        ref(obs.project_id, ctx, cls=Project)
        for ident in set(obs.subject_ids) | content_refs(obs):
            ref(ident, ctx, obs.project_id, obs.observed_at)
        check(content_refs(obs) <= set(obs.subject_ids), f'{ctx}: content references must be included in subject_ids')
        for ident in getattr(obs.content, 'issue_ids', []):
            ref(ident, ctx, cls=Issue)
        primary = getattr(obs.content, 'inspection_id', getattr(obs.content, 'activity_id', None))
        if primary:
            activity = ref(primary, ctx, cls=Activity)
            if isinstance(obs, InspectionObservation):
                check(isinstance(activity, Activity) and activity.attributes.activity_type == 'inspection', f'{ctx}: inspection observation needs inspection Activity')
        if isinstance(obs, PhotoObservation) and obs.content.artifact_id:
            artifact = ref(obs.content.artifact_id, ctx, cls=Artifact)
            check(isinstance(artifact, Artifact) and artifact.attributes.artifact_type == 'photo', f'{ctx}: photo artifact required')
        for ident in obs.source_event_ids:
            ev = event_ref(ident, ctx, obs.project_id)
            if not ev:
                continue
            check(ev.occurred_at <= obs.observed_at, f'{ctx}: observation precedes source event')
            check(ev.subject_id in obs.subject_ids, f'{ctx}: missing source subject')
            if isinstance(obs, InspectionObservation):
                check(ev.event_type in {'INSPECTION_FAILED', 'INSPECTION_PASSED'} and ev.subject_id == primary
                      and ev.payload.result == obs.content.result
                      and set(obs.content.issue_ids) <= set(ev.payload.issue_ids), f'{ctx}: inspection content/source mismatch')
                check(obs.content.finding == ('label_missing' if obs.content.result == 'fail' else 'label_present'), f'{ctx}: inconsistent finding/result')
            elif isinstance(obs, MessageObservation):
                expected = 'CORRECTION_REQUESTED' if obs.content.message_kind == 'correction_request' else 'INSPECTION_FAILED'
                check(ev.event_type == expected and set(obs.content.issue_ids) <= set(ev.payload.issue_ids), f'{ctx}: message content/source mismatch')
            elif isinstance(obs, ProgressObservation):
                check(ev.subject_id == primary and ev.event_type in {'ACTIVITY_STARTED', 'CORRECTION_STARTED', 'ACTIVITY_COMPLETED', 'CORRECTION_COMPLETED'}, f'{ctx}: progress/source mismatch')
            elif isinstance(obs, (PhotoObservation, ReportObservation)):
                check(ev.subject_id == primary and ev.event_type == 'CORRECTION_COMPLETED', f'{ctx}: completion record/source mismatch')
                if isinstance(obs, PhotoObservation):
                    check(set(obs.content.issue_ids) <= set(ev.payload.issue_ids), f'{ctx}: photo issue/source mismatch')
    for gt in d.ground_truth:
        ctx = gt.ground_truth_id
        ref(gt.project_id, ctx, cls=Project)
        origin = event_ref(gt.root_event_id, ctx, gt.project_id)
        if origin:
            check(origin.occurred_at <= gt.evaluation_as_of, f'{ctx}: workflow origin is in future')
        for ident in gt.affected_entity_ids:
            ref(ident, ctx, gt.project_id)
        for ident in gt.future_event_ids:
            event_ref(ident, ctx, gt.project_id)
        check(set(gt.future_event_ids) == {e.event_id for e in d.events if e.project_id == gt.project_id and e.occurred_at > gt.evaluation_as_of}, f'{ctx}: future_event_ids must be complete and strictly future')
        edges = []
        for ident in gt.event_link_ids:
            link = links.get(ident)
            check(link is not None, f'{ctx}: missing EventLink {ident}')
            if link:
                check(link.project_id == gt.project_id, f'{ctx}: cross-project EventLink')
                if link.relation_type != 'ASSOCIATED_WITH':
                    edges.append((link.source_event_id, link.target_event_id))
        if isinstance(gt.final_outcome, InspectionOutcome):
            final = event_ref(gt.final_outcome.event_id, ctx, gt.project_id)
            if final:
                check(final.payload.result == gt.final_outcome.result, f'{ctx}: final outcome mismatch')
            reachable = {gt.root_event_id}
            for _ in events:
                reachable |= {b for a, b in edges if a in reachable}
            check(gt.final_outcome.event_id in reachable, f'{ctx}: no workflow path to final outcome')
            for ident in gt.affected_entity_ids:
                if isinstance(entities.get(ident), Issue):
                    check(statuses.get(ident) == gt.final_outcome.issue_status, f'{ctx}: final Issue status mismatch')
    # Resolve provenance references and JSON pointers, never project these to AI.
    for row in [r for table, _ in groups for r in getattr(d, table)] + [d.metadata]:
        for attr in [row.provenance, *row.provenance.field_attributions.values()]:
            for ident in attr.evidence_ids:
                check(ident in d.metadata.evidence_register, f'unknown evidence {ident}')
            for ident in attr.assumption_ids:
                check(ident in d.metadata.assumption_register, f'unknown assumption {ident}')
            check(attr.basis not in {'evidence', 'inferred', 'mixed'} or bool(attr.evidence_ids), 'evidence/inferred/mixed requires evidence references')
            check(attr.basis != 'hypothesis' or bool(attr.assumption_ids), 'hypothesis requires assumption references')
        for pointer in row.provenance.field_attributions:
            try:
                if not pointer.startswith('/') or pointer.startswith('/provenance'):
                    raise KeyError(pointer)
                obj = row.model_dump()
                for key in pointer[1:].split('/'):
                    key = key.replace('~1', '/').replace('~0', '~')
                    obj = obj[int(key)] if isinstance(obj, list) else obj[key]
            except (KeyError, IndexError, ValueError, TypeError):
                errors.append(f'invalid provenance field pointer {pointer}')
    from .validation_schedule import validate_extensions
    errors.extend(validate_extensions(d))
    errors.extend(validate_commercial(d))
    if errors:
        raise ValidationError(errors)
