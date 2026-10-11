"""Exact rational outcome branching; no random source or sampled corpus."""
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from .v13 import Coordinates, execute, target, probabilities, support


@dataclass(frozen=True)
class Atom:
    coordinates: Coordinates
    ra: int
    rb: int
    mass: Fraction
    y: int
    actual: dict


def enumerate_atoms(regime, opportunity):
    for q in support():
        pa,pb=probabilities(regime,opportunity,x=q.x,h=q.h)
        for ra,rb in product((0,1),repeat=2):
            mass=(pa if ra else 1-pa)*(pb if rb else 1-pb)/128
            if not mass: continue
            actual=execute(q,ra,rb)
            yield Atom(q,ra,rb,mass,target(q,actual),actual)


def buckets(rows, projection):
    groups={}
    for row in rows:
        cell=groups.setdefault(projection(row),[Fraction(0),Fraction(0)])
        cell[0]+=row['mass'];cell[1]+=row['mass']*row['y']
    return groups


def posteriors(groups):
    return {k:positive/mass for k,(mass,positive) in groups.items()}


def brier(groups):
    return sum((mass*(positive/mass)*(1-positive/mass) for mass,positive in groups.values()),Fraction(0))


def accuracy(groups):
    return sum((max(positive,mass-positive) for mass,positive in groups.values()),Fraction(0))
