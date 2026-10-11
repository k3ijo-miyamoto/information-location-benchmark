"""Deterministic seed derivation used by the published benchmark."""
from hashlib import sha256

PREFIX = "construction-world-simulator|"
MASK = (1 << 63) - 1

def derive(namespace: str) -> int:
    return int.from_bytes(
        sha256((PREFIX + namespace).encode("utf-8")).digest()[:8],
        "big",
    ) & MASK
