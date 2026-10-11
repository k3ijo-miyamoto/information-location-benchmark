"""Exact counter-addressed route-and-retain sampling and lossless atom catalog."""
from collections import Counter
from dataclasses import asdict
from datetime import timedelta
from fractions import Fraction
from hashlib import sha256
from ..generator import v13 as g
from ..generator.v13_projection import normalize
from .native_stage_v_common_v13 import digest, CELLS


def uniform(seed, draw, purpose, bound):
    if not isinstance(bound, int) or not 0 < bound <= 2**256 or seed < 0 or draw < 0:
        raise ValueError('invalid counter/bound')
    base = f'construction-world-simulator|g13/stage_v/draw|{seed}|{draw}|{purpose}'
    limit = 2**256 - (2**256 % bound)
    retry = 0
    while True:
        address = base if retry == 0 else base + f'|reject|{retry}'
        value = int.from_bytes(sha256(address.encode('utf-8')).digest(), 'big')
        if value < limit:
            return value % bound
        retry += 1


def bernoulli(seed, draw, purpose, probability):
    if not isinstance(probability, Fraction) or not 0 <= probability <= 1:
        raise ValueError('exact Fraction probability required')
    return int(uniform(seed, draw, purpose, probability.denominator) < probability.numerator)


def sample(seed, cell, count=8192, *, start_draw=0, accepted_offset=0):
    if cell not in CELLS or count < 0 or start_draw < 0 or accepted_offset < 0:
        raise ValueError('invalid sampling request')
    regime, opportunity = cell.split('-')
    support = g.support()
    rows = []
    draw = start_draw
    while len(rows) < count:
        index = uniform(seed, draw, 'coordinate', len(support))
        q = support[index]
        if g.split(q) == 'train':
            pa, pb = g.probabilities(regime, opportunity, x=q.x, h=q.h)
            ra = bernoulli(seed, draw, 'ra', pa)
            rb = bernoulli(seed, draw, 'rb', pb)
            rows.append(dict(accepted_index=accepted_offset + len(rows), route_draw_index=draw,
                             public_support_index=index, atom_id=4*index+2*ra+rb,
                             ra=ra, rb=rb, y=g.target(q, g.execute(q, ra, rb))))
        draw += 1
    return rows, draw


def catalog():
    atoms, publics = [], []
    for index, q in enumerate(g.support()):
        public_hash = None
        for ra in (0, 1):
            for rb in (0, 1):
                data = g.build_world(q, ra, rb)  # Canonical validation inside builder and AI view.
                public = data.build_ai_view(g.ORIGIN + timedelta(minutes=120))
                normalized = normalize(public)
                current = digest(normalized)
                if public_hash is None:
                    public_hash = current
                    publics.append(public)
                if current != public_hash:
                    raise ValueError('realization leakage')
                events = {e.event_id: e for e in data.events}
                original = next(w for w in data.schedule_revisions[0].assignments if w.activity_id == 'D')
                y = int((events['ENDD'].occurred_at - original.planned_end).total_seconds() >= 7200)
                if y != g.target(q, g.execute(q, ra, rb)):
                    raise ValueError('serialized label mismatch')
                atoms.append(dict(atom_id=4*index+2*ra+rb, public_support_index=index,
                                  coordinates=asdict(q), ra=ra, rb=rb, y=y, split=g.split(q),
                                  canonical_sha256=digest(data.model_dump(mode='json')),
                                  normalized_public_sha256=current, public_template=index))
    if len(set(a['normalized_public_sha256'] for a in atoms)) != 128:
        raise ValueError('public support mismatch')
    return dict(version='stage-v-atom-catalog-0.1', atoms=atoms), publics


def counts(rows):
    return dict(by_group=dict(sorted(Counter(str(r['public_support_index']) for r in rows).items())),
                by_y=dict(sorted(Counter(str(r['y']) for r in rows).items())))
