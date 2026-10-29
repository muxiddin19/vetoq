"""Exp 1 / E18 -- semantic GROUP BY accuracy on Hard-2000 (2026-10-19).
Reported in paper_icde.md §6.10, Table 9; log in gpu_out/exp1_groupby.txt.

Replaces the earlier mock table (hard-coded numbers).  Data are generated
exactly as in E13 (hard_bench.make_clean / corrupt, same RNG call order), so
the rows are identical to E13 for each seed.

Query:  SELECT country, COUNT(*) FROM R GROUP BY country
Each method canonicalizes `country`; its classes are the groups.  Ground truth
is the clean country of each row.

Metrics (per seed x noise rate; table reports the mean over seeds):
  FM-LA   look-alike pairs (of 4) falsely merged: some group holds rows of both
  FM-all  distinct true-country pairs (of 66) that share some group
  SynRec  rows whose dirty country is a synonym (USA, UK, ...) placed in the
          plurality group of their true country
  P/R/F1  pairwise co-grouping precision / recall / F1 over row pairs
  CntErr  mean relative error of COUNT(*) reported for each true country
          (size of its plurality group vs. true count)

Settings fixed before running: ε = hard_bench.EPS (0.20), seeds 0-4,
noise rates hard_bench.RATES, 2000 rows, context attribute = capital.

Usage:  python run_first.py [--seeds 0 1 2 3 4] [--model MINILM] [--out exp1.csv]
"""

import argparse
import random
import time
from itertools import combinations

import numpy as np
import pandas as pd

from hard_bench import COUNTRIES, SYNONYMS, RATES, EPS, make_clean, corrupt, canon, norm

LOOKALIKES = [("Iran", "Iraq"), ("Austria", "Australia"), ("Slovakia", "Slovenia"), ("Niger", "Nigeria")]

METHODS = {
    "Exact SQL": dict(backend="exact"),
    "Normalize + Exact": dict(backend="norm"),
    "Edit-Distance": dict(backend="edit"),
    "Vector-Only": dict(backend="emb"),
    "Hybrid (no veto)": dict(backend="hybrid"),
    "Hybrid + Context Veto (raw, Def 8)": dict(backend="hybrid", ctx_col="capital", ctx_mode="hard"),
    "VetoQ (Hybrid + Pre-canon Context Veto)": dict(backend="hybrid", ctx_col="capital", ctx_mode="canon"),
    "Hybrid + Soft Veto (theta=0.5, E17)": dict(backend="hybrid", ctx_col="capital", ctx_mode="soft"),
}


def group_labels(df, model, backend, ctx_col=None, ctx_mode="hard"):
    if backend == "exact":
        return df["country"].astype(str)
    if backend == "norm":
        return df["country"].map(norm)
    return canon(df, "country", backend, model, ctx_col, ctx_mode)


def _pairs(x):
    x = np.asarray(x, dtype=np.int64)
    return int((x * (x - 1) // 2).sum())


def score(truth, dirty, pred):
    """truth: clean country per row; dirty: dirty country string; pred: group label."""
    ct = pd.crosstab(truth, pred)
    tp = _pairs(ct.values)
    pp = _pairs(ct.sum(axis=0).values)
    tpairs = _pairs(ct.sum(axis=1).values)
    p = tp / pp if pp else 1.0
    r = tp / tpairs if tpairs else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0

    shared = {c: set(ct.columns[ct.loc[c].values > 0]) for c in ct.index}
    merged = {(a, b) for a, b in combinations(sorted(ct.index), 2) if shared[a] & shared[b]}
    fm_la = sum(tuple(sorted(pr)) in merged for pr in LOOKALIKES)

    anchor = ct.idxmax(axis=1)                       # plurality group per true country
    gsize = ct.sum(axis=0)
    tsize = ct.sum(axis=1)
    cnt_err = float(np.mean([abs(gsize[anchor[c]] - tsize[c]) / tsize[c] for c in ct.index]))

    syn = np.array([d in SYNONYMS.get(t, ()) for t, d in zip(truth, dirty)])
    if syn.any():
        hit = np.array([pred.iloc[i] == anchor[truth.iloc[i]] for i in np.flatnonzero(syn)])
        syn_rec = float(hit.mean())
    else:
        syn_rec = np.nan
    return dict(FM_LA=fm_la, FM_all=len(merged), SynRec=syn_rec, P=p, R=r, F1=f1, CntErr=cnt_err)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--out", default=None, help="optional CSV of per-seed rows")
    args = ap.parse_args()
    np.random.seed(0)

    rows = []
    times = {m: 0.0 for m in METHODS}
    for seed in args.seeds:
        rng = random.Random(seed)
        clean = make_clean(rng)
        for rate in RATES:
            raw = corrupt(rng, clean, rate).astype(str)
            truth = clean["country"].reset_index(drop=True)
            for m, kw in METHODS.items():
                t0 = time.perf_counter()
                pred = group_labels(raw, args.model, **kw).reset_index(drop=True)
                times[m] += time.perf_counter() - t0
                rows.append(dict(seed=seed, rate=rate, method=m, **score(truth, raw["country"], pred)))
        print(f"[seed {seed}] done", flush=True)

    res = pd.DataFrame(rows)
    if args.out:
        res.to_csv(args.out, index=False)

    n_pairs = len(COUNTRIES) * (len(COUNTRIES) - 1) // 2
    print(f"\nExp 1 -- GROUP BY country on Hard-2000: seeds={args.seeds}, eps={EPS}, "
          f"emb={args.model}, context=capital")
    print(f"FM-LA of {len(LOOKALIKES)} look-alike pairs, FM-all of {n_pairs} pairs; mean over seeds\n")
    agg = res.groupby(["rate", "method"], sort=False)[["FM_LA", "FM_all", "SynRec", "P", "R", "F1", "CntErr"]].mean()
    for rate in RATES:
        print(f"--- noise rate {rate:.0%} ---")
        print(agg.loc[rate].to_string(float_format=lambda v: f"{v:.3f}"))
        print()
    print("--- overall (all rates) ---")
    overall = res.groupby("method", sort=False)[["FM_LA", "FM_all", "SynRec", "P", "R", "F1", "CntErr"]].mean()
    overall["time(s)"] = pd.Series(times)
    print(overall.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
