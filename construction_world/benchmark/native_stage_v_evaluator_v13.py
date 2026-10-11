"""Exact held-out evaluator. Posterior metadata never enters training/input caches."""
from fractions import Fraction as F
from functools import lru_cache
from math import isfinite, log
from ..generator import v13 as g
from ..generator.v13_exact import enumerate_atoms


@lru_cache(maxsize=16)
def posterior(cell, partition):
    regime, opportunity = cell.split('-')
    result = {}
    for index, q in enumerate(g.support()):
        if g.split(q) != partition:
            continue
        pa, pb = g.probabilities(regime, opportunity, x=q.x, h=q.h)
        p = sum(((pa if ra else 1-pa)*(pb if rb else 1-pb)*g.target(q, g.execute(q, ra, rb))
                 for ra in (0, 1) for rb in (0, 1)), F(0))
        result[index] = p
    # Independently cross-check the historical exact enumerator.
    reference = {}
    indices = {q:i for i,q in enumerate(g.support())}
    for atom in enumerate_atoms(regime, opportunity):
        index = indices[atom.coordinates]
        reference[index] = reference.get(index, F(0)) + atom.mass*128*atom.y
    if any(p != reference[i] for i, p in result.items()) or len(result) != 32:
        raise ValueError('exact posterior cross-check failed')
    return result


def metrics(predictions, probabilities, opportunity):
    if len(predictions) != 32 or len(probabilities) != 32:
        raise ValueError('32 uniformly weighted groups required')
    if any(not isfinite(q) or not 0 <= q <= 1 for q in predictions):
        raise ValueError('invalid probability prediction')
    ps = [float(p) for p in probabilities]
    brier = sum(p*(1-q)**2+(1-p)*q*q for p, q in zip(ps, predictions))/32
    bayes = sum(p*(1-p) for p in ps)/32
    regret = sum((q-p)**2 for p, q in zip(ps, predictions))/32
    gain = float(81*g.K[opportunity]**2)
    if abs((brier-bayes)-regret) > 1e-12 or abs(.25-bayes-gain) > 1e-12:
        raise ValueError('Brier/Bayes decomposition mismatch')
    recovery = (.25-brier)/gain
    if recovery > 1+1e-10 or brier-bayes < -1e-12:
        raise ValueError('invalid regret/recovery; never clip')
    bins = [[] for _ in range(10)]
    for p, q in zip(ps, predictions):
        bins[min(int(q*10), 9)].append(q-p)
    clipped = [min(1-1e-12, max(1e-12, q)) for q in predictions]
    return dict(brier=brier, bayes_brier=bayes, bayes_regret=regret,
                normalized_recovery=recovery, normalized_regret=regret/gain,
                expected_log_loss=-sum(p*log(q)+(1-p)*log(1-q) for p,q in zip(ps,clipped))/32,
                calibration_bias=sum(q-p for p,q in zip(ps,predictions))/32,
                posterior_mae=sum(abs(q-p) for p,q in zip(ps,predictions))/32,
                exact_ece_10=sum(abs(sum(values)) for values in bins)/32)
