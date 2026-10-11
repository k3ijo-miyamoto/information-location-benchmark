"""Reviewed scenario requirements, separate from reusable commercial invariants."""
from .models import FullAcceptance, PartialAcceptance, CommercialOutcome
from .commercial import RECORD_TABLES, COMMERCIAL_IDS
from .projection import project_balances, is_settled


def validate_scenario(d):
    errors = []
    def check(ok, message):
        if not ok: errors.append(message)
    for project in (e.entity_id for e in d.entities if e.entity_type == 'project'):
        rows = {t: [r for r in getattr(d, t) if r.project_id == project] for t in RECORD_TABLES}
        for t in RECORD_TABLES:
            if t != 'commercial_changes':
                check(bool(rows[t]), f'{project}: reviewed scenario requires {t}')
        check(len({r.estimate_id for r in rows['estimates']}) == 1, f'{project}: reviewed scenario requires one estimate identity')
        check(len({r.budget_id for r in rows['execution_budgets']}) == 1, f'{project}: reviewed scenario requires one budget identity')
        check(len({r.estimate_revision_id for r in rows['execution_budgets']}) == 1, f'{project}: reviewed scenario keeps estimate reference')
        truth = [g for g in d.ground_truth if g.project_id == project]
        check(len(truth) == 1, f'{project}: scenario requires one commercial Outcome')
        activities = {a for o in rows['orders'] for a in o.activity_ids}
        for g in truth:
            check(isinstance(g.final_outcome, CommercialOutcome), f'{project}: commercial outcome/profile mismatch')
            check(activities <= set(g.affected_entity_ids), f'{project}: missing affected commercial Activity')
            origin = next((e for e in d.events if e.event_id == g.root_event_id), None)
            if d.metadata.scenario == 'commercial_additional_work':
                allowed = {a for c in rows['commercial_changes'] for a in c.activity_ids}
            else:
                allowed = activities
            check(origin is not None and origin.event_type == 'ACTIVITY_CREATED' and origin.subject_id in allowed,
                  f'{project}: invalid scenario workflow origin')
        if d.metadata.scenario == 'commercial_additional_work':
            budgets = {b.revision_id: b for b in rows['execution_budgets']}
            for change in rows['commercial_changes']:
                base = budgets.get(change.base_budget_revision_id)
                revised = [b for b in rows['execution_budgets'] if b.change_id == change.change_id]
                if base and len(revised) == 1:
                    check(revised[0].planned_cost_amount.amount - base.planned_cost_amount.amount == sum(o.amount.amount for o in rows['orders'] if o.change_id == change.change_id), f'{project}: scenario budget delta/additional commitment mismatch')
            check(bool(rows['commercial_changes']), f'{project}: additional scenario requires change')
            check(any(b.version > 1 for b in rows['execution_budgets']), f'{project}: additional scenario requires revised budget')
            check(any(o.original_order_id for o in rows['orders']), f'{project}: additional scenario requires additional order')
            check(all(isinstance(a, FullAcceptance) for a in rows['acceptances']), f'{project}: additional scenario requires full acceptance')
        else:
            check(not rows['commercial_changes'] and all(b.version == 1 for b in rows['execution_budgets']) and
                  all(o.original_order_id is None and o.change_id is None for o in rows['orders']),
                  f'{project}: progress scenario excludes additional-work records')
            check(all(isinstance(a, PartialAcceptance) for a in rows['acceptances']), f'{project}: progress scenario requires partial acceptance')
            for t in ('acceptances', 'invoices', 'invoice_assessments', 'payments'):
                check(len(rows[t]) >= 2, f'{project}: progress scenario requires multiple {t}')
            for order in rows['orders']:
                check(sum(any(a.order_id == order.order_id for a in r.order_allocations) for r in rows['acceptances']) >= 2,
                      f'{order.order_id}: progress scenario requires multiple partial acceptances')
                check(sum(any(a.order_id == order.order_id for a in r.order_allocations) for r in rows['invoices']) >= 2,
                      f'{order.order_id}: progress scenario requires multiple invoices')
    return errors


def future_records(d, project, cutoff):
    return sorted((t, getattr(r, COMMERCIAL_IDS[t])) for t in RECORD_TABLES for r in getattr(d,t)
                  if r.project_id == project and (r.valid_from > cutoff or r.known_at > cutoff))


def validate_outcomes(d):
    errors = []
    for gt in d.ground_truth:
        out = gt.final_outcome
        if not isinstance(out, CommercialOutcome):
            continue  # scenario validation already rejects this
        def check(ok, message):
            if not ok: errors.append(f'{gt.ground_truth_id}: {message}')
        check(out.state_at_evaluation.as_of == gt.evaluation_as_of, 'evaluation cutoff mismatch')
        check(out.final_state.as_of == gt.created_at and gt.evaluation_as_of <= gt.created_at, 'final cutoff mismatch')
        records = [r for t in RECORD_TABLES for r in getattr(d,t) if r.project_id == gt.project_id]
        check(all(r.valid_from <= gt.created_at and r.known_at <= gt.created_at for r in records), 'Outcome before commercial facts')
        for snapshot in (out.state_at_evaluation, out.final_state):
            expected = project_balances(d, snapshot.as_of).get(gt.project_id)
            actual = snapshot.balances.model_dump(mode='json') if snapshot.balances is not None else None
            check(actual == expected, 'commercial snapshot mismatch')
        actual_future = [(r.table, r.record_id) for r in out.future_records]
        check(len(actual_future) == len(set(actual_future)) and sorted(actual_future) == future_records(d,gt.project_id,gt.evaluation_as_of), 'future commercial records mismatch')
        final = project_balances(d, gt.created_at).get(gt.project_id)
        check(out.final_settled == is_settled(final), 'settlement flag mismatch')
        # Full settlement is a requirement of these two reviewed end-to-end scenarios.
        check(is_settled(final), 'reviewed scenario requires final settlement')
    return errors
