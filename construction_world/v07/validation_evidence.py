"""Typed delivery evidence and explicit single-stream assessment chains."""
from .models import DeliveryWarningObservation, DeliveryUpdateObservation, Activity


def validate_evidence(data):
    errors=[]
    def check(ok,message):
        if not ok: errors.append(message)
    entities={e.entity_id:e for e in data.entities}
    plans={p.delivery_plan_id:p for p in data.delivery_plans}
    observations={o.observation_id:o for o in data.observations}
    events={e.event_id:e for e in data.events}
    warnings=[o for o in data.observations if isinstance(o,(DeliveryWarningObservation, DeliveryUpdateObservation))]

    def active_known(row,effective,known):
        return (row.valid_from<=effective and row.known_at<=known
                and (getattr(row,'valid_to',None) is None or effective<row.valid_to))

    def reachable(material,project,effective,known):
        relations=[r for r in data.relationships if active_known(r,effective,known)
                   and r.source_id in entities and entities[r.source_id].project_id==project]
        scope={r.source_id for r in relations if r.relation_type=='REQUIRES_MATERIAL' and r.target_id==material}
        for _ in entities:
            scope|={r.target_id for r in relations if r.relation_type=='PRECEDES' and r.source_id in scope}
            scope|={r.source_id for r in relations if r.relation_type=='DEPENDS_ON' and r.target_id in scope}
        return scope

    for warning in warnings:
        label=warning.observation_id
        evidence_known = warning.observed_at if isinstance(warning, DeliveryUpdateObservation) else warning.recorded_at
        check(data.metadata.profile=='schedule-material-v0.7',f'{label}: warning requires schedule profile')
        plan=plans.get(warning.content.delivery_plan_id)
        check(plan is not None,f'{label}: missing warning DeliveryPlan')
        check(len(warning.subject_ids)==len(set(warning.subject_ids)),f'{label}: duplicate warning subjects')
        if plan:
            check(plan.project_id==warning.project_id,f'{label}: cross-project warning DeliveryPlan')
            check(plan.valid_from<=warning.observed_at and plan.known_at<=warning.observed_at,
                  f'{label}: warning before referenced plan is available')
            check(plan.material_id in warning.subject_ids,f'{label}: warning must include plan Material')
            for ident in warning.subject_ids:
                entity=entities.get(ident)
                check(entity is not None and entity.project_id==warning.project_id and
                      active_known(entity,warning.observed_at,evidence_known),f'{label}: warning subject unavailable')
                check(ident==plan.material_id or isinstance(entity,Activity),f'{label}: inconsistent warning Material/subject')
                if isinstance(entity,Activity):
                    check(ident in reachable(plan.material_id,warning.project_id,warning.observed_at,evidence_known),
                          f'{label}: unrelated warning Activity')
            for ident in warning.source_event_ids:
                event=events.get(ident)
                if isinstance(warning, DeliveryUpdateObservation):
                    check(event is not None and event.project_id==warning.project_id
                          and event.recorded_at<=warning.observed_at, f'{label}: update source Event not known in this project')
                    continue
                check(event is not None and event.event_type in ('MATERIAL_AVAILABLE','MATERIAL_UNAVAILABLE')
                      and event.payload.delivery_plan_id==plan.delivery_plan_id and event.project_id==warning.project_id
                      and event.recorded_at<=warning.observed_at,
                      f'{label}: optional warning evidence must be known material Event for this plan')
    for risk in data.risk_assessments:
        check(len(risk.supporting_observation_ids)==len(set(risk.supporting_observation_ids)),f'{risk.risk_assessment_id}: duplicate observation evidence')
        for ident in risk.supporting_observation_ids:
            obs=observations.get(ident)
            check(isinstance(obs,(DeliveryWarningObservation, DeliveryUpdateObservation)),f'{risk.risk_assessment_id}: observation evidence must be existing DeliveryWarning or DeliveryUpdate (no cyclic risk evidence)')
            if not isinstance(obs,(DeliveryWarningObservation, DeliveryUpdateObservation)):continue
            check(obs.project_id==risk.project_id,f'{risk.risk_assessment_id}: cross-project observation evidence')
            check(obs.recorded_at<=risk.assessed_at,f'{risk.risk_assessment_id}: warning not known by assessed_at')
            plan=plans.get(obs.content.delivery_plan_id)
            if plan:
                check(set(risk.target_activity_ids)<=reachable(plan.material_id,risk.project_id,risk.assessed_at,risk.assessed_at),
                      f'{risk.risk_assessment_id}: warning/risk Activity scope mismatch')
        # Preserve legacy Event-only evidence timing. New mixed assessments use all evidence by assessed_at.
        if risk.supporting_observation_ids:
            check(all(i in events and events[i].recorded_at<=risk.assessed_at for i in risk.supporting_event_ids),
                  f'{risk.risk_assessment_id}: mixed Event evidence not known by assessed_at')
    if data.metadata.scenario in ('schedule_material_early_warning', 'schedule_multi_material_branching'):
        initial_warnings = [o for o in warnings if isinstance(o, DeliveryWarningObservation)]
        check(bool(initial_warnings),'early-warning scenario requires warning')
        check(any(o.content.delivery_plan_id in plans and o.observed_at<plans[o.content.delivery_plan_id].expected_at for o in initial_warnings),
              'early-warning scenario requires pre-deadline warning')
        supported=[r for r in data.risk_assessments if r.supporting_observation_ids]
        check(bool(supported),'early-warning scenario requires observation-supported risk')
        check(any(p.version>1 and any(r.project_id==p.project_id and r.known_at<=p.valid_from for r in supported)
                  for p in data.schedule_revisions),'early-warning scenario requires subsequent revised schedule')
    # Explicit evidence references only; no edges inferred from clock proximity.
    evidence_edges = {o.observation_id:set(o.source_event_ids) for o in data.observations}
    evidence_edges.update({r.risk_assessment_id:set(r.supporting_event_ids+r.supporting_observation_ids) for r in data.risk_assessments})
    evidence_edges.update({e.event_id:set(e.payload.evidence_event_ids) | ({e.payload.risk_assessment_id} if e.payload.risk_assessment_id else set()) for e in data.events})
    for update in (o for o in data.observations if isinstance(o, DeliveryUpdateObservation)):
        pending=list(evidence_edges[update.observation_id]); visited=set()
        while pending:
            ident=pending.pop()
            if ident==update.observation_id:
                check(False, f'{ident}: cyclic delivery evidence')
                break
            if ident not in visited:
                visited.add(ident);pending.extend(evidence_edges.get(ident,set()))
    risks = {r.risk_assessment_id:r for r in data.risk_assessments}
    successors = {}
    def stream(risk):
        evidence = [observations.get(i) for i in risk.supporting_observation_ids]
        if not evidence or not all(isinstance(o, (DeliveryWarningObservation, DeliveryUpdateObservation)) for o in evidence):
            return set()
        return {o.content.delivery_plan_id for o in evidence}
    for risk in data.risk_assessments:
        previous = risk.previous_assessment_id
        if previous is None:
            continue
        label = risk.risk_assessment_id
        predecessor = risks.get(previous)
        check(previous != label, f'{label}: self supersession')
        check(previous not in successors, f'{label}: forked supersession chain')
        successors[previous] = label
        check(predecessor is not None, f'{label}: missing predecessor')
        if predecessor is None:
            continue
        for field in ('project_id', 'risk_kind', 'assessor_kind', 'baseline_revision_id'):
            check(getattr(predecessor,field)==getattr(risk,field), f'{label}: supersession {field} mismatch')
        check(set(predecessor.target_activity_ids)==set(risk.target_activity_ids), f'{label}: supersession target set mismatch')
        check(predecessor.assessed_at < risk.assessed_at, f'{label}: supersession requires strictly later assessment')
        check(predecessor.known_at <= risk.assessed_at, f'{label}: predecessor not known by assessment')
        before, after = stream(predecessor), stream(risk)
        check(len(before)==len(after)==1 and before==after, f'{label}: supersession requires same single delivery stream')
        visited = {label}
        cursor = previous
        while cursor in risks:
            if cursor in visited:
                check(False, f'{label}: cyclic supersession chain')
                break
            visited.add(cursor)
            cursor = risks[cursor].previous_assessment_id
    if data.metadata.scenario == 'schedule_multi_material_branching':
        check(any(isinstance(o, DeliveryUpdateObservation) for o in data.observations), 'branching scenario requires delivery update')
        check(any(r.previous_assessment_id for r in data.risk_assessments), 'branching scenario requires explicit supersession')
    return errors
