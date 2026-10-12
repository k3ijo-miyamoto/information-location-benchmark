#!/usr/bin/env python3
"""Fail if publication tree contains known private-authoring/company-origin strings."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
# Construct sensitive tokens without storing them contiguously in this checker itself.
needles=[
    "AND"+"PAD",
    "d"+"SPACE",
    "/tmp/"+"andpad",
    "One"+"Drive",
    "Chat"+"GPT",
    "Co"+"dex",
]
skip={Path(__file__).resolve()}
bad=[]
for p in ROOT.rglob("*"):
    if not p.is_file() or p.resolve() in skip or ".git" in p.parts:
        continue
    if p.suffix.lower() in {".png",".pdf",".gz",".pt",".zip"}:
        continue
    try: text=p.read_text(encoding="utf-8",errors="strict")
    except (UnicodeDecodeError,OSError): continue
    for n in needles:
        if n.lower() in text.lower(): bad.append((str(p.relative_to(ROOT)),n))
if bad:
    raise SystemExit("publication cleanliness failure: "+repr(bad))
print("PASS: publication cleanliness scan")
