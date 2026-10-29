"""E-C (formula vs execution) and preliminary E-A (ER metrics vs operator errors) on Hard-2000.

Usage: python exp_t1/run_ec_ea.py [--seeds 0 1 2 3 4] [--out gpu_out/t1_ec_ea.csv]
"""
import argparse
import itertools
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hard_bench import RATES, corrupt, make_clean          # noqa: E402
from run_first import METHODS, group_labels                 # noqa: E402
import contingency as C                                      # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def one_table(seed, rate, raw, clean, rng_np):
    truth = clean["country"].to_numpy()
    n_rows = len(raw)
    amount = rng_np.lognormal(mean=3.0, sigma=1.0, size=n_rows).round(2)   # synthetic SUM attribute
    side = rng_np.random(n_rows) < 0.5                                        # R / S split for the join
    exact_rows = {e: int(np.flatnonzero((raw["country"].to_numpy() == e) & (truth == e))[0])
                  for e in np.unique(truth) if ((raw["country"].to_numpy() == e) & (truth == e)).any()}
    rows, mism = [], Counter_()
    for m, kw in METHODS.items():
        pred = group_labels(raw, MODEL, **kw).astype(str).to_numpy()
        n = C.table(pred, truth)
        ms = C.masses(n)
        # Lemma 1 / 2
        f_cnt = ms["m"] - ms["s"]
        x_cnt = C.execute_groupby(pred, truth)
        mism["COUNT"] += int((f_cnt.sort_index() - x_cnt.sort_index()).abs().gt(1e-9).sum())
        n_w = pd.crosstab(pd.Series(pred, name="c"), pd.Series(truth, name="e"),
                          values=amount, aggfunc="sum").fillna(0.0).reindex(index=n.index, columns=n.columns)
        f_sum = C.formula_groupby_sum(n_w, ms)
        x_sum = C.execute_groupby(pred, truth, amount)
        mism["SUM"] += int((f_sum.sort_index() - x_sum.sort_index()).abs().gt(1e-6).sum())
        # Lemma 3
        f_cd, _ = C.formula_count_distinct(n)
        x_cd = C.execute_count_distinct(pred, truth)
        mism["COUNT_DISTINCT"] += int(f_cd != x_cd)
        # Lemma 4
        nR, nS = C.table(pred[side], truth[side]), C.table(pred[~side], truth[~side])
        f_j = C.formula_join(nR, nS)
        x_j = C.execute_join(pred[side], truth[side], pred[~side], truth[~side])
        mism["JOIN"] += int(f_j != x_j)
        # Lemma 6
        f_s = C.formula_select(n, pred, exact_rows)
        x_s = C.execute_select(pred, truth, exact_rows)
        mism["SELECT"] += int(f_s != x_s)
        er = C.er_metrics(n)
        op = C.operator_errors(n)
        jp = f_j["TP"] / (f_j["TP"] + f_j["FP"]) if f_j["TP"] + f_j["FP"] else 1.0
        jr = f_j["TP"] / (f_j["TP"] + f_j["FN"]) if f_j["TP"] + f_j["FN"] else 1.0
        true_sum = pd.Series({e: amount[truth == e].sum() for e in np.unique(truth)})
        rows.append(dict(seed=seed, rate=rate, method=m, **er, **op,
                         SUM_err=float((f_sum.abs() / true_sum).mean()),
                         JOIN_FP_rate=1 - jp, JOIN_FN_rate=1 - jr))
    return rows, mism


class Counter_(dict):
    def __missing__(self, k):
        return 0


def inversions(res, metric, higher_better, op):
    """Fraction of method pairs (within a table) where the ER metric and the operator error disagree."""
    agree = disagree = 0
    for _, g in res.groupby(["seed", "rate"]):
        for (_, a), (_, b) in itertools.combinations(g.iterrows(), 2):
            dm = (a[metric] - b[metric]) * (1 if higher_better else -1)   # >0: a better by metric
            do = b[op] - a[op]                                             # >0: a better by operator
            if abs(dm) < 1e-12 or abs(do) < 1e-12:
                continue
            if (dm > 0) == (do > 0):
                agree += 1
            else:
                disagree += 1
    return disagree / max(1, agree + disagree), agree + disagree


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--out", default="gpu_out/t1_ec_ea.csv")
    args = ap.parse_args()
    allrows, total = [], Counter_()
    for seed in args.seeds:
        rng = random.Random(seed)
        rng_np = np.random.default_rng(seed)
        clean = make_clean(rng)
        for rate in RATES:
            raw = corrupt(rng, clean, rate).astype(str)
            rows, mism = one_table(seed, rate, raw, clean.astype(str), rng_np)
            allrows += rows
            for k, v in mism.items():
                total[k] += v
        print(f"[seed {seed}] done", flush=True)
    res = pd.DataFrame(allrows)
    res.to_csv(args.out, index=False)
    n_checks = len(res)
    print(f"\n=== E-C: formula vs execution, {n_checks} (table, method) cases ===")
    for k in ["COUNT", "SUM", "COUNT_DISTINCT", "JOIN", "SELECT"]:
        print(f"  {k:<15} mismatches: {total[k]}")
    print("\n=== E-A (preliminary, Hard-2000): share of method pairs where ER metric and operator error disagree ===")
    ops = ["COUNT_err", "SUM_err", "CD_err", "JOIN_FP_rate", "JOIN_FN_rate"]
    print(f"{'metric':<8}" + "".join(f"{o:>14}" for o in ops))
    for metric, hb in [("pairF1", True), ("B3F1", True), ("VI", False), ("BMD", False)]:
        cells = []
        for o in ops:
            frac, cnt = inversions(res, metric, hb, o)
            cells.append(f"{frac:>9.3f} ({cnt})")
        print(f"{metric:<8}" + "".join(f"{c:>14}" for c in cells))
    print("\n=== mean per method ===")
    cols = ["pairF1", "B3F1", "VI", "BMD", "M", "S", "COUNT_err", "SUM_err", "CD_err", "JOIN_FP_rate", "JOIN_FN_rate"]
    print(res.groupby("method", sort=False)[cols].mean().to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
