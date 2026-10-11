"""Profile boundaries and planning/material contracts; invoked by core validator."""
from .models import (Activity, Issue, Material, ScheduleOutcome, InspectionOutcome,
                     MaterialObservation, ScheduleObservation, RiskObservation, DeliveryWarningObservation)

SCHEDULE_EVENTS = {'MATERIAL_UNAVAILABLE', 'MATERIAL_AVAILABLE', 'BLOCKING_STARTED', 'BLOCKING_CLEARED',
                   'SCHEDULE_REVISED', 'RISK_ASSESSED', 'SCHEDULE_ISSUE_RESOLVED'}


def validate_extensions(d):
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    entities = {e.entity_id: e for e in d.entities}
    events = {e.event_id: e for e in d.events}
    plans = {p.schedule_revision_id: p for p in d.schedule_revisions}
    deliveries = {p.delivery_plan_id: p for p in d.delivery_plans}
    blocks = {b.blocking_condition_id: b for b in d.blocking_conditions}
    risks = {r.risk_assessment_id: r for r in d.risk_assessments}
    def ref(table, ident, project, label):
        row = table.get(ident)
        check(row is not None, f'{label}: missing reference {ident}')
        if row:
            check(row.project_id == project, f'{label}: cross-project reference {ident}')
        return row
    def entity(ident, project, cls, label):
        e = ref(entities, ident, project, label)
        check(isinstance(e, cls), f'{label}: wrong entity type {ident}')
        return e
    def before(a, b):
        return (a.occurred_at, a.sequence) < (b.occurred_at, b.sequence)
    def event_matches(kind, field, ident):
        return [e for e in d.events if e.event_type == kind and getattr(e.payload, field) == ident]
    inspection = d.metadata.profile == 'inspection-v0.7'
    if inspection:
        check(not any((d.schedule_revisions, d.delivery_plans, d.blocking_conditions, d.risk_assessments)), 'inspection profile cannot carry schedule/material records')
        check(not any(isinstance(e, Material) for e in d.entities), 'inspection profile cannot carry Material')
        check(not any(e.event_type in SCHEDULE_EVENTS for e in d.events), 'inspection profile cannot carry schedule events')
        check(not any(isinstance(o, (MaterialObservation, ScheduleObservation, RiskObservation, DeliveryWarningObservation)) for o in d.observations), 'inspection profile observation mismatch')
    else:
        check(not any(e.event_type.startswith(('INSPECTION_', 'CORRECTION_', 'REINSPECTION_')) for e in d.events), 'schedule profile must not fabricate inspection events')
        check(all(o.observation_type in {'progress_record', 'material_availability', 'schedule_revision', 'schedule_risk', 'delivery_warning', 'delivery_update'} for o in d.observations), 'schedule profile observation mismatch')
    for e in d.events:
        payload = e.payload.model_dump(exclude_none=True)
        allowed = {'result', 'issue_ids'}
        required = {
            'MATERIAL_UNAVAILABLE': 'delivery_plan_id', 'MATERIAL_AVAILABLE': 'delivery_plan_id',
            'BLOCKING_STARTED': 'blocking_condition_id', 'BLOCKING_CLEARED': 'blocking_condition_id',
            'SCHEDULE_REVISED': 'schedule_revision_id', 'RISK_ASSESSED': 'risk_assessment_id',
        }.get(e.event_type)
        if required:
            allowed.add(required)
            check(getattr(e.payload, required) is not None, f'{e.event_id}: missing typed payload {required}')
        if e.event_type in {'BLOCKING_STARTED', 'BLOCKING_CLEARED', 'SCHEDULE_ISSUE_RESOLVED'}:
            allowed.add('evidence_event_ids')
            check(bool(e.payload.evidence_event_ids), f'{e.event_id}: resolution/condition requires evidence')
        check(not {k for k, v in payload.items() if v is not None and v != []} - allowed,
              f'{e.event_id}: payload fields do not belong to event type')
        for ident in e.payload.evidence_event_ids:
            source = ref(events, ident, e.project_id, e.event_id)
            if source:
                check(before(source, e), f'{e.event_id}: future/self evidence event')
        subject = entities.get(e.subject_id)
        if e.event_type in {'ISSUE_RESOLVED', 'ISSUE_CLOSED', 'ISSUE_REOPENED'}:
            check(isinstance(subject, Issue) and subject.attributes.issue_kind in {'inspection_finding', 'quality'}, f'{e.event_id}: inspection lifecycle requires inspection/quality Issue')
        if e.event_type == 'SCHEDULE_ISSUE_RESOLVED':
            check(isinstance(subject, Issue) and subject.attributes.issue_kind == 'schedule', f'{e.event_id}: schedule resolution requires schedule Issue')
            check(any(events[x].event_type == 'BLOCKING_CLEARED' and e.subject_id in events[x].payload.issue_ids
                      for x in e.payload.evidence_event_ids if x in events), f'{e.event_id}: schedule resolution requires related cleared blocking evidence')
    by_project = {}
    for p in d.schedule_revisions:
        by_project.setdefault(p.project_id, []).append(p)
        ids = [w.activity_id for w in p.assignments]
        check(len(ids) == len(set(ids)), f'{p.schedule_revision_id}: duplicate assignment')
        for w in p.assignments:
            a = entity(w.activity_id, p.project_id, Activity, p.schedule_revision_id)
            if a:
                check(a.valid_from <= p.valid_from and a.known_at <= p.known_at, f'{p.schedule_revision_id}: plan references future Activity')
        check(p.known_at >= p.valid_from, f'{p.schedule_revision_id}: plan known before revision is issued')
        if p.version == 1:
            check(p.previous_revision_id is None, 'baseline cannot replace prior revision')
        else:
            prev = ref(plans, p.previous_revision_id, p.project_id, p.schedule_revision_id)
            if prev:
                check(prev.version + 1 == p.version and prev.valid_from < p.valid_from and prev.known_at <= p.known_at,
                      f'{p.schedule_revision_id}: invalid monotonic revision chain')
                check({w.activity_id for w in prev.assignments} == set(ids), f'{p.schedule_revision_id}: preserve assignment scope')
            matches = event_matches('SCHEDULE_REVISED', 'schedule_revision_id', p.schedule_revision_id)
            check(len(matches) == 1 and matches[0].occurred_at == p.valid_from and matches[0].recorded_at == p.known_at,
                  f'{p.schedule_revision_id}: revision needs matching Event at valid/known times')
    for project, versions in by_project.items():
        check(sorted(p.version for p in versions) == list(range(1, len(versions)+1)), f'{project}: baseline/revisions must be preserved without branches or gaps')
    for e in d.events:
        if e.payload.schedule_revision_id:
            p = ref(plans, e.payload.schedule_revision_id, e.project_id, e.event_id)
            check(p is not None and p.version > 1, f'{e.event_id}: revision must reference nonbaseline')
    for p in d.delivery_plans:
        entity(p.material_id, p.project_id, Material, p.delivery_plan_id)
        check(p.valid_from <= p.known_at <= p.expected_at, f'{p.delivery_plan_id}: expected delivery is a prospective plan')
        arrivals = event_matches('MATERIAL_AVAILABLE', 'delivery_plan_id', p.delivery_plan_id)
        check(len(arrivals) <= 1, f'{p.delivery_plan_id}: partial/multiple arrival unsupported; use distinct plans')
        if arrivals:
            check(all(before(e, arrivals[0]) for e in event_matches('MATERIAL_UNAVAILABLE', 'delivery_plan_id', p.delivery_plan_id)),
                  f'{p.delivery_plan_id}: non-arrival reported after actual arrival')
    for e in d.events:
        if e.event_type.startswith('MATERIAL_'):
            plan = ref(deliveries, e.payload.delivery_plan_id, e.project_id, e.event_id)
            if plan:
                check(e.subject_id == plan.material_id and plan.valid_from <= e.occurred_at, f'{e.event_id}: invalid delivery subject/time')
                if e.event_type == 'MATERIAL_UNAVAILABLE':
                    check(e.occurred_at >= plan.expected_at, f'{e.event_id}: non-arrival cannot precede expected arrival')
    for b in d.blocking_conditions:
        entity(b.activity_id, b.project_id, Activity, b.blocking_condition_id)
        entity(b.material_id, b.project_id, Material, b.blocking_condition_id)
        plan = ref(deliveries, b.delivery_plan_id, b.project_id, b.blocking_condition_id)
        check(plan is not None and plan.material_id == b.material_id, f'{b.blocking_condition_id}: material/delivery mismatch')
        check(any(r.relation_type == 'REQUIRES_MATERIAL' and r.source_id == b.activity_id and r.target_id == b.material_id
                  and r.valid_from <= b.valid_from for r in d.relationships), f'{b.blocking_condition_id}: missing material requirement')
        starts = event_matches('BLOCKING_STARTED', 'blocking_condition_id', b.blocking_condition_id)
        ends = event_matches('BLOCKING_CLEARED', 'blocking_condition_id', b.blocking_condition_id)
        check(len(starts) == 1 and len(ends) <= 1, f'{b.blocking_condition_id}: condition requires one start, at most one clear')
        if starts:
            start = starts[0]
            check(start.subject_id == b.activity_id and start.occurred_at == b.valid_from and start.recorded_at == b.known_at,
                  f'{b.blocking_condition_id}: blocking start times/subject mismatch')
            check(any(events[x].event_type == 'MATERIAL_UNAVAILABLE' and events[x].payload.delivery_plan_id == b.delivery_plan_id
                      for x in start.payload.evidence_event_ids if x in events), f'{b.blocking_condition_id}: missing unavailable evidence')
            if ends:
                end = ends[0]
                check(end.subject_id == b.activity_id and before(start, end), f'{b.blocking_condition_id}: clear before start')
                check(any(events[x].event_type == 'MATERIAL_AVAILABLE' and events[x].payload.delivery_plan_id == b.delivery_plan_id
                          for x in end.payload.evidence_event_ids if x in events), f'{b.blocking_condition_id}: clear needs actual availability')
            for work in d.events:
                if work.subject_id == b.activity_id and work.event_type == 'ACTIVITY_STARTED' and not before(work, start):
                    check(bool(ends) and before(ends[0], work), f'{work.event_id}: activity starts while blocked')
    for e in d.events:
        if e.payload.blocking_condition_id:
            ref(blocks, e.payload.blocking_condition_id, e.project_id, e.event_id)
    for r in d.relationships:
        if r.relation_type == 'REQUIRES_MATERIAL':
            for plan in d.schedule_revisions:
                if plan.version != 1:
                    continue
                for window in plan.assignments:
                    if window.activity_id != r.source_id or r.valid_from > window.planned_start:
                        continue
                    facts = sorted([e for e in d.events if e.subject_id == r.target_id
                                    and e.event_type in {'MATERIAL_AVAILABLE', 'MATERIAL_UNAVAILABLE'}
                                    and e.occurred_at <= window.planned_start], key=lambda e: (e.occurred_at, e.sequence))
                    if facts and facts[-1].event_type == 'MATERIAL_UNAVAILABLE':
                        check(any(b.activity_id == r.source_id and b.material_id == r.target_id
                                  and b.valid_from <= window.planned_start for b in d.blocking_conditions),
                              f'{r.source_id}: unavailable required material at baseline start needs explicit blocking condition')
            for e in d.events:
                if e.subject_id == r.source_id and e.event_type == 'ACTIVITY_STARTED':
                    check(any(a.event_type == 'MATERIAL_AVAILABLE' and a.subject_id == r.target_id and before(a, e) for a in d.events), f'{e.event_id}: required material not available')
    for risk in d.risk_assessments:
        plan = ref(plans, risk.baseline_revision_id, risk.project_id, risk.risk_assessment_id)
        check(plan is not None and plan.version == 1, f'{risk.risk_assessment_id}: risk requires baseline')
        for ident in risk.target_activity_ids:
            entity(ident, risk.project_id, Activity, risk.risk_assessment_id)
            windows = [w for w in plan.assignments if w.activity_id == ident] if plan else []
            check(bool(windows) and risk.assessed_at <= windows[0].planned_start, f'{risk.risk_assessment_id}: pre-start risk assessment after baseline start')
        for ident in risk.supporting_event_ids:
            e = ref(events, ident, risk.project_id, risk.risk_assessment_id)
            if e:
                check(e.occurred_at <= risk.assessed_at and e.recorded_at <= risk.known_at, f'{risk.risk_assessment_id}: supporting evidence is future')
        matches = event_matches('RISK_ASSESSED', 'risk_assessment_id', risk.risk_assessment_id)
        check(len(matches) == 1 and matches[0].occurred_at == risk.assessed_at and matches[0].recorded_at == risk.known_at,
              f'{risk.risk_assessment_id}: matching assessment Event required')
    for e in d.events:
        if e.payload.risk_assessment_id:
            risk = ref(risks, e.payload.risk_assessment_id, e.project_id, e.event_id)
            check(risk is not None and e.subject_id in risk.target_activity_ids, f'{e.event_id}: risk subject mismatch')
    for o in d.observations:
        for ident in o.source_event_ids:
            e = events.get(ident)
            if not e:
                continue
            if isinstance(o, MaterialObservation):
                p = ref(deliveries, o.content.delivery_plan_id, o.project_id, o.observation_id)
                expected = 'MATERIAL_AVAILABLE' if o.content.availability == 'available' else 'MATERIAL_UNAVAILABLE'
                check(e.event_type == expected and e.subject_id == o.content.material_id and e.payload.delivery_plan_id == o.content.delivery_plan_id,
                      f'{o.observation_id}: material observation disagrees with world evidence')
            elif isinstance(o, ScheduleObservation):
                p = ref(plans, o.content.schedule_revision_id, o.project_id, o.observation_id)
                check(p is not None and set(o.content.activity_ids) == {w.activity_id for w in p.assignments}
                      and e.event_type == 'SCHEDULE_REVISED' and e.payload.schedule_revision_id == o.content.schedule_revision_id,
                      f'{o.observation_id}: schedule observation mismatch')
            elif isinstance(o, RiskObservation):
                risk = ref(risks, o.content.risk_assessment_id, o.project_id, o.observation_id)
                check(risk is not None and o.content.risk_level == risk.level and set(o.content.activity_ids) == set(risk.target_activity_ids)
                      and e.event_type == 'RISK_ASSESSED' and e.payload.risk_assessment_id == risk.risk_assessment_id,
                      f'{o.observation_id}: risk observation mismatch')
    for gt in d.ground_truth:
        if d.metadata.profile == 'commercial-v0.7':
            continue  # CommercialOutcome is checked by the commercial validator.
        check(isinstance(gt.final_outcome, InspectionOutcome) == inspection, f'{gt.ground_truth_id}: outcome/profile mismatch')
        if not isinstance(gt.final_outcome, ScheduleOutcome):
            continue
        outcome = gt.final_outcome
        target_ids = [x.activity_id for x in outcome.activities]
        check(len(target_ids) == len(set(target_ids)), f'{gt.ground_truth_id}: duplicate outcome activity')
        check(set(outcome.affected_downstream_activity_ids) <= set(target_ids) <= set(gt.affected_entity_ids), f'{gt.ground_truth_id}: invalid affected activities')
        edges = [(l.source_event_id, l.target_event_id) for l in d.event_links
                 if l.event_link_id in gt.event_link_ids and l.relation_type != 'ASSOCIATED_WITH']
        reachable = {gt.root_event_id}
        for _ in events:
            reachable |= {b for a, b in edges if a in reachable}
        for row in outcome.activities:
            entity(row.activity_id, gt.project_id, Activity, gt.ground_truth_id)
            start = ref(events, row.actual_start_event_id, gt.project_id, gt.ground_truth_id)
            end = ref(events, row.actual_end_event_id, gt.project_id, gt.ground_truth_id)
            check(row.actual_end_event_id in reachable, f'{gt.ground_truth_id}: outcome not reachable from origin')
            if not start or not end:
                continue
            check(start.subject_id == end.subject_id == row.activity_id and start.event_type == 'ACTIVITY_STARTED'
                  and end.event_type == 'ACTIVITY_COMPLETED' and before(start, end), f'{gt.ground_truth_id}: actual event mismatch')
            check(end.occurred_at <= gt.created_at, f'{gt.ground_truth_id}: realized outcome before actual completion')
            for rev_id, start_delta, end_delta, baseline in [
                (row.baseline_revision_id, row.start_deviation_minutes, row.end_deviation_minutes, True),
                (row.revised_revision_id, row.revised_start_deviation_minutes, row.revised_end_deviation_minutes, False)]:
                if rev_id is None:
                    check(start_delta is None and end_delta is None, f'{gt.ground_truth_id}: revised deltas without revision')
                    continue
                plan = ref(plans, rev_id, gt.project_id, gt.ground_truth_id)
                if not plan:
                    continue
                check((plan.version == 1) == baseline, f'{gt.ground_truth_id}: wrong baseline/revised version')
                window = next((w for w in plan.assignments if w.activity_id == row.activity_id), None)
                check(window is not None, f'{gt.ground_truth_id}: missing plan assignment')
                if window:
                    check((start.occurred_at - window.planned_start).total_seconds() == start_delta * 60 if start_delta is not None else False, f'{gt.ground_truth_id}: start deviation mismatch')
                    check((end.occurred_at - window.planned_end).total_seconds() == end_delta * 60 if end_delta is not None else False, f'{gt.ground_truth_id}: end deviation mismatch')
            check(row.realized_delay == (row.start_deviation_minutes > 0 or row.end_deviation_minutes > 0), f'{gt.ground_truth_id}: realized delay mismatch')
    from .validation_evidence import validate_evidence
    errors.extend(validate_evidence(d))
    return errors
