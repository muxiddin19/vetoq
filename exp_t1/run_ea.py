"""E-A: do ER quality metrics predict operator errors? (paper §VII)

For every dataset instance and every candidate equivalence (grouping.method_grid), compute
  ER metrics  : pairwise F1, B-cubed F1, VI, BMD (GMD with unit costs)
  op. errors  : COUNT per entity, SUM per entity, COUNT(DISTINCT), JOIN false-pair rate, JOIN missed-pair rate
Report (1) the share of method pairs whose order by an ER metric disagrees with their order by an operator error,
and (2) the regret of choosing the equivalence by an ER metric instead of by the operator error itself.

Usage: python -P exp_t1/run_ea.py [--quick] [--out gpu_out/t1_ea.csv]
"""
import argparse
import itertools
import os
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import contingency as C          # noqa: E402
import datasets as D             # noqa: E402
import grouping as G             # noqa: E402

OPS = ["COUNT_err", "SUM_err", "CD_err", "JOIN_FP_rate", "JOIN_FN_rate"]
METRICS = [("pairF1", True), ("B3F1", True), ("VI", False), ("BMD", False)]


def instances(quick):
    out = []
    hard_seeds = [0, 1] if quick else [0, 1, 2, 3, 4]
    for s in hard_seeds:
        for r in (0.1, 0.2, 0.3):
            out.append(("Hard-2000", lambda s=s, r=r: D.hard(s, r)))
    out.append(("Hospital", D.hospital))
    out.append(("Beers", D.beers))
    for s in ([0] if quick else [0, 1, 2]):
        out.append(("MusicBrainz", lambda s=s: D.musicbrainz(s, 300 if quick else 1000)))
        out.append(("Settlements", lambda s=s: D.settlements(s, 150 if quick else 400)))
    if not quick:
        out.append(("Affiliations", D.affiliations))
    return out


def evaluate(df, meta, inst_id):
    truth = df["E"].astype(str).to_numpy()
    rng = np.random.default_rng(zlib.crc32(inst_id.encode()))      # reproducible across runs
    amount = rng.lognormal(3.0, 1.0, len(df)).round(2)
    side = rng.random(len(df)) < 0.5
    true_sum = pd.Series(amount).groupby(truth).sum()
    rows = []
    G.clear_cache()
    for name, kw in G.method_grid(meta["ctx_kind"]).items():
        t0 = time.perf_counter()
        pred = np.asarray(G.group(df, ctx_kind=meta["ctx_kind"], **kw)).astype(str)
        dt = time.perf_counter() - t0
        n = C.table(pred, truth)
        ms = C.masses(n)
        er = C.er_metrics(n)
        op = C.operator_errors(n)
        n_w = pd.crosstab(pd.Series(pred, name="c"), pd.Series(truth, name="e"), values=amount,
                          aggfunc="sum").fillna(0.0).reindex(index=n.index, columns=n.columns)
        f_sum = C.formula_groupby_sum(n_w, ms)
        j = C.formula_join(C.table(pred[side], truth[side]), C.table(pred[~side], truth[~side]))
        jp = j["TP"] / (j["TP"] + j["FP"]) if j["TP"] + j["FP"] else 1.0
        jr = j["TP"] / (j["TP"] + j["FN"]) if j["TP"] + j["FN"] else 1.0
        rows.append(dict(dataset=meta["name"].split(" ")[0], inst=inst_id, method=name, **er, **op,
                         SUM_err=float((f_sum.abs() / true_sum.reindex(f_sum.index)).mean()),
                         JOIN_FP_rate=1 - jp, JOIN_FN_rate=1 - jr, n_rows=len(df),
                         n_entities=n.shape[1], n_classes=n.shape[0], sec=dt))
    return rows


def disagreement(res, metric, hb, op):
    agree = dis = 0
    for _, g in res.groupby("inst"):
        recs = g[[metric, op]].to_numpy()
        for (m1, o1), (m2, o2) in itertools.combinations(recs, 2):
            dm = (m1 - m2) * (1 if hb else -1)
            do = o2 - o1
            if abs(dm) < 1e-12 or abs(do) < 1e-12:
                continue
            if (dm > 0) == (do > 0):
                agree += 1
            else:
                dis += 1
    return dis / max(1, agree + dis)


def regret(res, metric, hb, op):
    """Mean over instances of (op error of the method the ER metric ranks best) - (best op error)."""
    out = []
    for _, g in res.groupby("inst"):
        pick = g.loc[g[metric].idxmax() if hb else g[metric].idxmin()]
        out.append(pick[op] - g[op].min())
    return float(np.mean(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(HERE), "gpu_out", "t1_ea.csv"))
    args = ap.parse_args()
    rows = []
    for ds, loader in instances(args.quick):
        df, meta = loader()
        t0 = time.perf_counter()
        rows += evaluate(df, meta, meta["name"])
        print(f"[{meta['name']}] rows={len(df)} entities={df['E'].nunique()} keys={df['K'].nunique()} "
              f"{time.perf_counter() - t0:.0f}s", flush=True)
        pd.DataFrame(rows).to_csv(args.out, index=False)
    res = pd.DataFrame(rows)

    print("\n=== E-A (1): share of method pairs where ER metric order disagrees with operator-error order ===")
    for scope, sub in [("ALL", res)] + list(res.groupby("dataset")):
        print(f"\n[{scope}]  instances={sub['inst'].nunique()}  methods/instance={sub.groupby('inst').size().max()}")
        print(f"{'metric':<8}" + "".join(f"{o:>14}" for o in OPS))
        for metric, hb in METRICS:
            print(f"{metric:<8}" + "".join(f"{disagreement(sub, metric, hb, o):>14.3f}" for o in OPS))

    print("\n=== E-A (2): regret of choosing the equivalence by an ER metric (mean over instances, ALL) ===")
    print(f"{'chooser':<10}" + "".join(f"{o:>14}" for o in OPS))
    for metric, hb in METRICS:
        print(f"{metric:<10}" + "".join(f"{regret(res, metric, hb, o):>14.4f}" for o in OPS))
    print("\n=== best method per operator (count of instances where it is best) ===")
    for o in OPS:
        best = res.loc[res.groupby("inst")[o].idxmin(), "method"].value_counts().head(5)
        print(f"{o:<14} " + ", ".join(f"{m} ({c})" for m, c in best.items()))


if __name__ == "__main__":
    main()
