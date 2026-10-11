"""Commercial-only contracts and safe projections; no accounting recognition."""
from collections import defaultdict
from .models import Activity, Project, CommercialOutcome, FullAcceptance, PartialAcceptance, ProgressObservation

COMMERCIAL_IDS = {
    'estimates': 'revision_id', 'execution_budgets': 'revision_id',
    'orders': 'order_id', 'commercial_changes': 'change_id',
    'acceptances': 'acceptance_id', 'invoices': 'invoice_id',
    'invoice_assessments': 'assessment_id', 'payments': 'payment_id',
    'commercial_observations': 'observation_id',
}
RECORD_TABLES = tuple(k for k in COMMERCIAL_IDS if k != 'commercial_observations')


def references(table, row):
    """Backward typed record references only; never store future effects on a change."""
    refs = []
    if table in {'estimates', 'execution_budgets'} and row.previous_revision_id:
        refs.append((table, row.previous_revision_id))
    if table == 'execution_budgets':
        refs.append(('estimates', row.estimate_revision_id))
        if row.change_id:
            refs.append(('commercial_changes', row.change_id))
    elif table == 'orders':
        refs.append(('execution_budgets', row.budget_revision_id))
        if row.original_order_id:
            refs.append(('orders', row.original_order_id))
        if row.change_id:
            refs.append(('commercial_changes', row.change_id))
    elif table == 'commercial_changes':
        refs.append(('execution_budgets', row.base_budget_revision_id))
        refs.extend(('orders', x) for x in row.base_order_ids)
    elif table == 'acceptances':
        refs.extend(('orders', x.order_id) for x in row.order_allocations)
    elif table == 'invoices':
        refs.extend(('orders', x.order_id) for x in row.order_allocations)
        refs.extend(('acceptances', x) for x in row.acceptance_ids)
    elif table == 'invoice_assessments':
        refs.append(('invoices', row.invoice_id))
    elif table == 'payments':
        for x in row.invoice_allocations:
            refs.extend((('invoices', x.invoice_id), ('invoice_assessments', x.assessment_id)))
    return refs


def money_fields(table, row):
    if table == 'acceptances':
        return [x.accepted_amount for x in row.order_allocations]
    if table == 'estimates':
        return [row.total_amount]
    if table == 'execution_budgets':
        return [row.planned_cost_amount]
    if table == 'invoice_assessments':
        return [row.assessed_amount]
    if table in {'orders', 'invoices', 'payments'}:
        return [row.amount] + [x.amount for x in getattr(row, 'order_allocations', ())] + [x.amount for x in getattr(row, 'invoice_allocations', ())]
    return []


def validate_commercial(d):
    errors = []
    def check(ok, message):
        if not ok:
            errors.append(message)
    commercial = d.metadata.profile == 'commercial-v0.7'
    if not commercial:
        check(not any(getattr(d, t) for t in COMMERCIAL_IDS), 'non-commercial profile cannot carry commercial records')
        check(not any(isinstance(g.final_outcome, CommercialOutcome) for g in d.ground_truth), 'commercial outcome/profile mismatch')
        return errors

    # Strict reviewed scenario boundary, not a universal real-world workflow.
    check(not any((d.schedule_revisions, d.delivery_plans, d.blocking_conditions, d.risk_assessments)), 'commercial profile cannot carry schedule/material records')
    check(all(e.entity_type in {'project', 'actor', 'activity', 'artifact', 'location'} for e in d.entities), 'commercial profile cannot carry Material or Issue')
    check(all(not isinstance(e, Activity) or e.attributes.activity_type == 'installation' for e in d.entities), 'commercial profile only supports installation Activities')
    allowed_events = {'PROJECT_CREATED', 'PROJECT_STARTED', 'ACTIVITY_CREATED', 'ACTIVITY_STARTED', 'ACTIVITY_COMPLETED'}
    check(all(e.event_type in allowed_events for e in d.events), 'commercial profile rejects inspection/material/monetary Event shortcuts')
    check(all(o.observation_type == 'progress_record' for o in d.observations), 'commercial profile cannot encode invoices in ReportObservation')
    check(all(not e.payload.issue_ids and e.payload.result is None for e in d.events), 'commercial physical Event cannot carry inspection semantics')
    entities = {e.entity_id: e for e in d.entities}
    events = {e.event_id: e for e in d.events}
    maps = {t: {getattr(r, COMMERCIAL_IDS[t]): r for r in getattr(d, t)} for t in RECORD_TABLES}
    projects = {e.entity_id for e in d.entities if isinstance(e, Project)}
    record_ids = {getattr(r, COMMERCIAL_IDS[t]) for t in RECORD_TABLES for r in getattr(d, t)}
    identity_owners = {}
    all_existing_ids = set(entities) | set(events) | record_ids | {r.relationship_id for r in d.relationships} | {o.observation_id for o in d.observations} | {o.observation_id for o in d.commercial_observations} | {g.ground_truth_id for g in d.ground_truth} | {l.event_link_id for l in d.event_links}
    dimensions = defaultdict(set)

    def ref(table, ident, row, label):
        target = maps[table].get(ident)
        check(target is not None, f'{label}: missing {table} reference {ident}')
        if target:
            check(target.project_id == row.project_id, f'{label}: cross-project reference {ident}')
            check(target.valid_from <= row.valid_from and target.known_at <= row.known_at, f'{label}: future reference {ident}')
        return target

    def unique(values, label):
        check(len(values) == len(set(values)), f'{label}: duplicate reference')

    for table in RECORD_TABLES:
        for row in getattr(d, table):
            label = getattr(row, COMMERCIAL_IDS[table])
            project = entities.get(row.project_id)
            check(isinstance(project, Project), f'{label}: missing Project')
            if isinstance(project, Project):
                check(project.valid_from <= row.valid_from and project.known_at <= row.known_at, f'{label}: future Project')
            unique(row.activity_ids, label)
            for ident in row.activity_ids:
                a = entities.get(ident)
                check(isinstance(a, Activity) and a.project_id == row.project_id, f'{label}: invalid Activity {ident}')
                if isinstance(a, Activity):
                    check(a.valid_from <= row.valid_from and a.known_at <= row.known_at, f'{label}: future Activity')
                    check(a.valid_to is None or row.valid_from < a.valid_to, f'{label}: inactive Activity')
            for target_table, ident in references(table, row):
                ref(target_table, ident, row, label)
            for money in money_fields(table, row):
                dimensions[row.project_id].add((money.currency, money.tax_basis))
    for project, dims in dimensions.items():
        check(len(dims) == 1, f'{project}: incompatible money currency/tax basis')

    # Stable identities are distinct from revision/Entity IDs; histories cannot branch.
    for table, stable_field in [('estimates', 'estimate_id'), ('execution_budgets', 'budget_id')]:
        histories = defaultdict(list)
        for row in getattr(d, table):
            stable_id = getattr(row, stable_field)
            owner = (table, row.project_id)
            check(stable_id not in all_existing_ids, f'{stable_id}: stable identity collides with record ID')
            check(identity_owners.get(stable_id, owner) == owner, f'{stable_id}: cross-project/type identity')
            identity_owners[stable_id] = owner
            histories[(row.project_id, stable_id)].append(row)
        for (project, stable_id), history in histories.items():
            history.sort(key=lambda x: x.version)
            check([r.version for r in history] == list(range(1, len(history)+1)), f'{stable_id}: preserve baseline and consecutive versions')
            for i, row in enumerate(history):
                expected = history[i-1].revision_id if i else None
                check(row.previous_revision_id == expected, f'{row.revision_id}: invalid previous revision chain')
                if i:
                    prev = history[i-1]
                    check(prev.valid_from < row.valid_from and prev.known_at <= row.known_at, f'{row.revision_id}: nonmonotonic revision times')
                if table == 'execution_budgets':
                    check((row.change_id is None) == (row.version == 1), f'{row.revision_id}: revised budget requires CommercialChange')
                    if row.change_id and row.change_id in maps['commercial_changes']:
                        change = maps['commercial_changes'][row.change_id]
                        check(change.base_budget_revision_id == row.previous_revision_id, f'{row.revision_id}: change must refer to preceding budget')

    for order in d.orders:
        check(bool(order.original_order_id) == bool(order.change_id), f'{order.order_id}: additional order requires original and change')
        if order.original_order_id:
            original = maps['orders'].get(order.original_order_id)
            if original:
                check(original.original_order_id is None and original.order_id != order.order_id, f'{order.order_id}: must preserve original order')
                check(original.valid_from < order.valid_from, f'{order.order_id}: original order must precede additional')
            change = maps['commercial_changes'].get(order.change_id)
            budget = maps['execution_budgets'].get(order.budget_revision_id)
            if change:
                check(order.original_order_id in change.base_order_ids, f'{order.order_id}: unrelated original order')
                check(set(change.activity_ids) <= set(order.activity_ids), f'{order.order_id}: additional Activity mismatch')
            if budget:
                check(budget.change_id == order.change_id, f'{order.order_id}: budget/change mismatch')
        else:
            budget = maps['execution_budgets'].get(order.budget_revision_id)
            if budget:
                check(budget.version == 1, f'{order.order_id}: original order needs baseline budget')

    for change in d.commercial_changes:
        unique(change.base_order_ids, change.change_id)
        check(bool(change.activity_ids), f'{change.change_id}: additional_work needs physical Activity')
        revisions = [r for r in d.execution_budgets if r.change_id == change.change_id]
        orders = [o for o in d.orders if o.change_id == change.change_id]
        check(len(revisions) == 1 and bool(orders), f'{change.change_id}: requires revised budget and additional order')
        base = maps['execution_budgets'].get(change.base_budget_revision_id)
        base_orders = [maps['orders'][x] for x in change.base_order_ids if x in maps['orders']]
        for order in base_orders:
            check(order.budget_revision_id == change.base_budget_revision_id, f'{change.change_id}: base order/budget mismatch')
        check(not set(change.activity_ids) & {a for o in base_orders for a in o.activity_ids}, f'{change.change_id}: additional work must not replace original Activity')
        if base and len(revisions) == 1:
            check(set(change.activity_ids) <= set(revisions[0].activity_ids), f'{change.change_id}: revised budget must include additional Activity')

    accepted = defaultdict(int)
    full_orders = set()
    acceptance_counts = defaultdict(int)
    observations = {o.observation_id: o for o in d.observations}
    for acceptance in d.acceptances:
        label = acceptance.acceptance_id
        order_ids = [x.order_id for x in acceptance.order_allocations]
        unique(order_ids, label)
        activities = {a for x in order_ids if x in maps['orders'] for a in maps['orders'][x].activity_ids}
        check(bool(acceptance.activity_ids) and set(acceptance.activity_ids) <= activities, f'{label}: acceptance Activity coverage mismatch')
        for allocation in acceptance.order_allocations:
            accepted[allocation.order_id] += allocation.accepted_amount.amount
            acceptance_counts[allocation.order_id] += 1
            check(allocation.accepted_amount.amount > 0, f'{label}: acceptance must be positive')
        if isinstance(acceptance, FullAcceptance):
            unique(acceptance.completion_event_ids, label)
            completed = set()
            for ident in acceptance.completion_event_ids:
                event = events.get(ident)
                check(event is not None, f'{label}: missing completion Event')
                if event:
                    check(event.project_id == acceptance.project_id and event.event_type == 'ACTIVITY_COMPLETED', f'{label}: requires physical completion')
                    check(event.occurred_at <= acceptance.valid_from and event.recorded_at <= acceptance.known_at, f'{label}: acceptance before completion known/effective')
                    completed.add(event.subject_id)
            check(completed == activities and bool(activities) and set(acceptance.activity_ids) == activities, f'{label}: completion coverage mismatch')
            for allocation in acceptance.order_allocations:
                full_orders.add(allocation.order_id)
                order = maps['orders'].get(allocation.order_id)
                if order:
                    check(allocation.accepted_amount == order.amount, f'{label}: full acceptance must equal commitment')
        else:
            unique(acceptance.progress_observation_ids, label)
            covered = set()
            for ident in acceptance.progress_observation_ids:
                obs = observations.get(ident)
                check(isinstance(obs, ProgressObservation), f'{label}: missing progress evidence')
                if isinstance(obs, ProgressObservation):
                    check(obs.project_id == acceptance.project_id, f'{label}: cross-project progress')
                    check(obs.observed_at <= acceptance.valid_from and obs.recorded_at <= acceptance.known_at, f'{label}: future progress evidence')
                    check(obs.visibility == 'ai_visible', f'{label}: progress evidence must be public')
                    check(obs.content.reported_progress > 0, f'{label}: positive progress evidence required')
                    check(all(i in events and events[i].recorded_at <= acceptance.known_at for i in obs.source_event_ids), f'{label}: progress source not yet known')
                    covered.add(obs.content.activity_id)
            check(covered == set(acceptance.activity_ids), f'{label}: progress Activity coverage mismatch')
    for ident, amount in accepted.items():
        if ident in maps['orders']:
            check(amount <= maps['orders'][ident].amount.amount, f'{ident}: cumulative accepted above commitment')
        check(ident not in full_orders or acceptance_counts[ident] == 1, f'{ident}: full acceptance cannot mix with other acceptances')

    invoiced = defaultdict(int)
    for invoice in d.invoices:
        order_ids = [x.order_id for x in invoice.order_allocations]
        unique(order_ids, invoice.invoice_id)
        unique(invoice.acceptance_ids, invoice.invoice_id)
        check(sum(x.amount.amount for x in invoice.order_allocations) == invoice.amount.amount, f'{invoice.invoice_id}: invoice allocation total mismatch')
        cited = [maps['acceptances'][a] for a in invoice.acceptance_ids if a in maps['acceptances']]
        for allocation in invoice.order_allocations:
            ident = allocation.order_id
            invoiced[ident] += allocation.amount.amount
            available = sum(x.accepted_amount.amount for a in cited for x in a.order_allocations if x.order_id == ident)
            check(allocation.amount.amount <= available, f'{invoice.invoice_id}: invoice exceeds cited acceptance')
            # Prefix ceilings in BOTH time axes; future acceptance cannot justify an earlier invoice.
            for clock in ('valid_from', 'known_at'):
                cutoff = getattr(invoice, clock)
                ceiling = sum(x.accepted_amount.amount for a in d.acceptances if getattr(a, clock) <= cutoff for x in a.order_allocations if x.order_id == ident)
                total = sum(x.amount.amount for i in d.invoices if getattr(i, clock) <= cutoff for x in i.order_allocations if x.order_id == ident)
                check(total <= ceiling, f'{invoice.invoice_id}: cumulative invoiced above accepted at {clock}')
    for ident, amount in invoiced.items():
        if ident in maps['orders']:
            check(amount <= maps['orders'][ident].amount.amount, f'{ident}: invoiced above commitment; amendment required')
    assessment_invoices = []
    for assessment in d.invoice_assessments:
        assessment_invoices.append(assessment.invoice_id)
        invoice = maps['invoices'].get(assessment.invoice_id)
        if invoice:
            check(assessment.assessed_amount.amount <= invoice.amount.amount, f'{assessment.assessment_id}: assessment exceeds invoice')
    unique(assessment_invoices, 'invoice assessments')
    paid = defaultdict(int)
    for payment in d.payments:
        unique([x.invoice_id for x in payment.invoice_allocations], payment.payment_id)
        check(sum(x.amount.amount for x in payment.invoice_allocations) == payment.amount.amount, f'{payment.payment_id}: payment allocation total mismatch')
        for allocation in payment.invoice_allocations:
            approval = maps['invoice_assessments'].get(allocation.assessment_id)
            if approval:
                check(approval.invoice_id == allocation.invoice_id, f'{payment.payment_id}: approval/invoice mismatch')
                paid[allocation.assessment_id] += allocation.amount.amount
    for ident, amount in paid.items():
        check(amount <= maps['invoice_assessments'][ident].assessed_amount.amount, f'{ident}: cumulative payment exceeds approved amount')

    for notice in d.commercial_observations:
        target = maps[notice.target_kind].get(notice.target_id)
        check(target is not None, f'{notice.observation_id}: missing notice target')
        if target:
            check(target.project_id == notice.project_id, f'{notice.observation_id}: cross-project notice')
            check(target.valid_from <= notice.observed_at and target.known_at <= notice.recorded_at, f'{notice.observation_id}: future target in notice')

    from .scenarios import validate_scenario, validate_outcomes
    errors.extend(validate_scenario(d))
    # Evaluate projections only after referential/money checks: invalid input must not crash helpers.
    if not errors:
        errors.extend(validate_outcomes(d))

    return errors


PUBLIC_FIELDS = {
    'estimates': ('revision_id', 'estimate_id', 'version', 'previous_revision_id', 'total_amount'),
    'execution_budgets': ('revision_id', 'budget_id', 'version', 'previous_revision_id', 'estimate_revision_id', 'planned_cost_amount', 'change_id'),
    'orders': ('order_id', 'amount', 'budget_revision_id', 'counterparty_role', 'original_order_id', 'change_id'),
    'commercial_changes': ('change_id', 'category', 'base_budget_revision_id', 'base_order_ids'),
    'acceptances': ('acceptance_id', 'acceptance_kind', 'order_allocations'),
    'invoices': ('invoice_id', 'amount', 'order_allocations', 'acceptance_ids'),
    'invoice_assessments': ('assessment_id', 'invoice_id', 'decision', 'assessed_amount'),
    'payments': ('payment_id', 'amount', 'invoice_allocations'),
}


def commercial_view(d, at, known=True):
    entities = {e.entity_id for e in d.entities if e.valid_from <= at and (not known or e.known_at <= at)
                and (e.valid_to is None or e.valid_to > at or (known and e.valid_to_known_at > at))}
    selected = {t: [r for r in getattr(d, t) if r.valid_from <= at and (not known or r.known_at <= at)
                    and {r.project_id, *r.activity_ids} <= entities] for t in RECORD_TABLES}
    event_ids = {e.event_id for e in d.events if e.occurred_at <= at and (not known or e.recorded_at <= at)
                 and {e.project_id, e.subject_id, *e.actor_ids} <= entities}
    observation_ids = {o.observation_id for o in d.observations if o.visibility == 'ai_visible' and o.observed_at <= at and (not known or o.recorded_at <= at) and {o.project_id, *o.subject_ids} <= entities and set(o.source_event_ids) <= event_ids}
    # Dependency closure: a record cannot reveal a reference whose contents are unknown.
    while True:
        ids = {t: {getattr(r, COMMERCIAL_IDS[t]) for r in rows} for t, rows in selected.items()}
        filtered = {t: [r for r in rows if all(ident in ids[k] for k, ident in references(t, r))
                       and (t != 'acceptances' or (set(r.completion_event_ids) <= event_ids if isinstance(r, FullAcceptance) else set(r.progress_observation_ids) <= observation_ids))] for t, rows in selected.items()}
        if sum(map(len, filtered.values())) == sum(map(len, selected.values())):
            break
        selected = filtered
    result = {t: [r.model_dump(mode='json', include={'project_id', 'activity_ids', 'valid_from', 'known_at', *PUBLIC_FIELDS[t]}) for r in rows]
              for t, rows in selected.items()}
    result['commercial_observations'] = [n.model_dump(mode='json', include={'observation_id', 'project_id', 'observation_type', 'target_kind', 'target_id', 'observed_at', 'recorded_at'})
        for n in d.commercial_observations if n.visibility == 'ai_visible' and n.observed_at <= at and n.recorded_at <= at and n.target_id in ids[n.target_kind]]
    return result
