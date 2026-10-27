#!/usr/bin/env python3
"""Restore the upstream files that as-run/ does not redistribute.

Four tools (scGAD, scSemiCluster, scSemiGAN, CIForm) ran from copies of their authors'
code, and their repositories publish no licence, so the copies are not included here.
upstream/UPSTREAM.tsv pins each file to the GitHub commit it was copied from, and
upstream/patches/ holds the lines the study changed, as zero-context unified diffs. This
script downloads each file, applies its patch, and checks the result against the sha256
of the file that ran (also listed in MANIFEST.tsv).

    python3 as-run/fetch_upstream.py

Python 3.8+, standard library only; needs network access to raw.githubusercontent.com.
The restored files are git-ignored.
"""
import csv
import hashlib
import re
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUNK = re.compile(rb"@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def split_lines(data):
    """Lines split on \\n only, each keeping its ending (CR bytes stay part of the line)."""
    return re.findall(rb"[^\n]*\n|[^\n]+\Z", data)


def apply_patch(base, patch):
    """Apply a zero-context unified diff (`diff -U0`) to the bytes of the pinned file."""
    src, lines = split_lines(base), split_lines(patch)
    i = 0
    while i < len(lines) and not lines[i].startswith(b"@@"):
        i += 1                                         # the ---/+++ header
    out, pos = [], 0
    while i < len(lines):
        m = HUNK.match(lines[i])
        if not m:
            raise ValueError(f"unexpected patch line: {lines[i][:60]!r}")
        i += 1
        start, n_old = int(m[1]), int(m[2] if m[2] is not None else 1)
        n_new = int(m[4] if m[4] is not None else 1)
        old, new = [], []
        for count, sign, acc in ((n_old, b"-", old), (n_new, b"+", new)):
            while len(acc) < count:
                if lines[i][:1] != sign:
                    raise ValueError(f"malformed hunk at {lines[i][:60]!r}")
                acc.append(lines[i][1:])
                i += 1
                if i < len(lines) and lines[i].startswith(b"\\"):   # "\ No newline at end of file"
                    acc[-1] = acc[-1][:-1]
                    i += 1
        at = start - 1 if n_old else start                # with no old lines, insert after line `start`
        if src[at:at + n_old] != old:
            raise ValueError(f"patch does not match the pinned file at line {start}")
        out += src[pos:at] + new
        pos = at + n_old
    return b"".join(out + src[pos:])


def main():
    with open(HERE / "upstream" / "UPSTREAM.tsv", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    failed = []
    for r in rows:
        url = f"https://raw.githubusercontent.com/{r['repository']}/{r['commit']}/{r['upstream_path']}"
        with urllib.request.urlopen(url, timeout=60) as resp:
            data = resp.read()
        if r["patch"]:
            data = apply_patch(data, (HERE / r["patch"]).read_bytes())
        out = HERE / r["path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(data)
        ok = hashlib.sha256(data).hexdigest() == r["sha256"]
        print(f"{'ok  ' if ok else 'FAIL'}  {r['path']}")
        if not ok:
            failed.append(r["path"])
    print(f"{len(rows) - len(failed)} of {len(rows)} files restored and matched the copies that ran.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
