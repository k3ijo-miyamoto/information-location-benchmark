"""Design arithmetic only: no models, sampling, optimizers or learner evaluation."""
from math import isfinite, sqrt

N_GRID = (128, 512, 2048, 8192)
R_GRID = (8, 12, 16, 24, 32)
# Rounded upward from Student-t ppf(1-.05/12, R-1); six co-primary contrasts.
CRITICAL = {8: 3.635807422, 12: 3.208122334, 16: 3.036283223,
            24: 2.886263784, 32: 2.818308158}
SD_INFLATION = 1.460972437  # sqrt(5 / chi2.ppf(.20, 5)), rounded upward


def parameter_components(family, hidden):
    linear = lambda a, b: (a + 1) * b
    d, fields, slots, edge_width = 2575, 73, 128, 4
    if family == 'tabular':
        return dict(content=linear(d, 8), endpoints=129*4,
                    reference_projection=linear(fields*4, 8),
                    fusion=linear(slots*16, hidden), hidden=linear(hidden, hidden),
                    output=linear(hidden, 1))
    if family not in ('static', 'temporal', 'static_size_control'):
        raise ValueError('unknown family')
    result = dict(content=linear(d, hidden), edge_types=2*fields*edge_width,
                  messages=2*(linear(hidden, hidden)+linear(hidden+edge_width, hidden)),
                  fusion=linear((3 if family == 'temporal' else 2)*hidden, hidden),
                  output=linear(hidden, 1))
    if family == 'temporal':
        result['gru'] = 6*hidden*hidden+6*hidden
    return result


def required_corpora(corpus_paired_sds):
    """Only six SDs enter; no observed effect, p value or model choice argument."""
    if len(corpus_paired_sds) != 6 or any(not isfinite(s) or s < 0 for s in corpus_paired_sds):
        raise ValueError('six finite nonnegative corpus-level SDs required')
    upper = max(corpus_paired_sds)*SD_INFLATION
    for r in R_GRID:
        if CRITICAL[r]*upper/sqrt(r) <= .10:
            return {'corpora': r, 'precision_feasible': True}
    return {'corpora': 32, 'precision_feasible': False}


def coverage(n):
    if n <= 0:
        raise ValueError('positive training size required')
    return dict(expected_groups=64*(1-(63/64)**n),
                missing_any_union_bound=min(1., 64*(63/64)**n),
                expected_labels_per_group=n/64,
                per_group_bernoulli_sd_bound=4/sqrt(n))


def run_budget(corpora, initializations=3):
    return dict(runs=8*len(N_GRID)*corpora*initializations*4,
                epochs=8*len(N_GRID)*corpora*initializations*4*100,
                example_presentations=8*corpora*initializations*4*100*sum(N_GRID),
                accepted_train_worlds=8*corpora*max(N_GRID),
                expected_route_draws=2*8*corpora*max(N_GRID))
