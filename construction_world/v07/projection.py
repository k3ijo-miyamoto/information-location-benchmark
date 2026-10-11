"""Derived, time-bounded commercial balances. No accounting recognition."""
from .commercial import commercial_view
from .models import CommercialBalances


def project_balances(data, at, known=True):
    """Internal validated-input projection; public entry point validates the Dataset."""
    view = commercial_view(data, at, known=known)
    result = {}
    for project in sorted({r['project_id'] for r in view['orders']}):
        rows = {t: [r for r in values if r['project_id'] == project] for t, values in view.items()}
        basis = rows['orders'][0]['amount']
        ordered = sum(r['amount']['amount'] for r in rows['orders'])
        accepted = sum(a['accepted_amount']['amount'] for r in rows['acceptances'] for a in r['order_allocations'])
        invoiced = sum(r['amount']['amount'] for r in rows['invoices'])
        assessed = sum(r['assessed_amount']['amount'] for r in rows['invoice_assessments'])
        paid = sum(r['amount']['amount'] for r in rows['payments'])
        values = dict(ordered_amount=ordered, accepted_amount=accepted, remaining_to_accept=ordered-accepted,
                      invoiced_amount=invoiced, remaining_to_invoice=ordered-invoiced,
                      accepted_uninvoiced_amount=accepted-invoiced, assessed_amount=assessed,
                      unassessed_amount=invoiced-assessed, paid_amount=paid, remaining_to_pay=assessed-paid,
                      invoice_unpaid_amount=invoiced-paid)
        result[project] = CommercialBalances(**{k: dict(amount=v, currency=basis['currency'], tax_basis=basis['tax_basis']) for k,v in values.items()}).model_dump(mode='json')
    return result


def commercial_projection(data, as_of, known=True):
    from .validation import validate_dataset
    from .views import _at
    validate_dataset(data)
    return project_balances(data, _at(as_of), known=known)


def is_settled(balances):
    return balances is not None and all(balances[k]['amount'] == 0 for k in
        ('remaining_to_accept', 'remaining_to_invoice', 'remaining_to_pay', 'invoice_unpaid_amount', 'unassessed_amount'))
