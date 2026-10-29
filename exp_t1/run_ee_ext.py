"""E-E with external candidates: the Hard-multi choosers of run_ed.py, with Splink outputs added to the pool.

Pool "base" = the 24 built-in candidates (as in run_ed.py); pool "ext" = base + 6 unsupervised Splink candidates
(EM; key only / key + capital; thresholds 0.5 / 0.9 / 0.99), computed on the same union of country values.
Label-trained Splink is excluded: it uses the ground truth and is not deployable.
For each pool: oracle, VetoQ-E(q) (witness = calling code on a fraction q of rows), and the Splink default alone.

Usage: python -P exp_t1/run_ee_ext.py [--seeds 0 1 2 3 4]
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import contingency as C                               # noqa: E402
import grouping as G                                   # noqa: E402
from hard_multi import generate                        # noqa: E402
from run_ed import QS, QUERIES, merge_frac, query_errors, union_frame   # noqa: E402
from run_eh import prepare, run_splink                 # noqa: E402

SPLINK_DEFAULT = "Splink-em-ctx-0.5"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()
    logging.getLogger("splink").setLevel(logging.ERROR)
    rows = []
    for seed in args.seeds:
        for rate in (0.1, 0.2, 0.3):
            countries, customers, orders = generate(seed, rate)
            u = union_frame(countries, customers, orders)
            truth = u["E"].astype(str).to_numpy()
            code = u["code"].astype(str).to_numpy()
            cont_true = dict(zip(countries["name"], countries["continent"]))
            cont_code = dict(zip(countries["code"].astype(str), countries["continent"]))
            rng = np.random.default_rng(seed * 100 + int(rate * 10))   # same draws as run_ed.py
            u_draw = rng.random(len(u))
            masks = {q: (u_draw < q) & (u["tbl"] != "D").to_numpy() for q in QS}
            full = (u["tbl"] != "D").to_numpy()
            cands = {}
            G.clear_cache()
            for name, kw in G.method_grid("str").items():
                cands[name] = ("base", np.asarray(G.group(u, ctx_kind="str", **kw)).astype(str))
            d = prepare(u, {"ctx_kind": "str"})
            t0 = time.perf_counter()
            for use_ctx in (False, True):
                labs, _ = run_splink(d, "str", use_ctx, "em")
                for th, lab in labs.items():
                    cands[f"Splink-em-{'ctx' if use_ctx else 'key'}-{th}"] = ("ext", lab)
            sec = time.perf_counter() - t0
            for name, (pool, lab) in cands.items():
                rec = dict(seed=seed, rate=rate, method=name, pool=pool,
                           **{f"true_{q}": v for q, v in query_errors(u, lab, truth, full, cont_true).items()})
                for q in QS:
                    est = query_errors(u, lab, code, masks[q], cont_code)
                    rec.update({f"est{q}_{k}": v for k, v in est.items()})
                    rec[f"estM_{q}"] = merge_frac(lab, code, masks[q])
                rows.append(rec)
            print(f"[seed {seed} rate {rate}] rows {len(u)}, Splink {sec:.0f}s", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(os.path.dirname(HERE), "gpu_out", "t1_ee_ext.csv"), index=False)
    report(res)


def report(res):
    inst = res.groupby(["seed", "rate"])

    def pick(g, chooser, q, query):
        if chooser == "oracle":
            return g[f"true_{query}"].min()
        if chooser == "VetoQ-E":
            return g.loc[C.choose(g, f"est{q}_{query}", f"estM_{q}"), f"true_{query}"]
        if chooser == "Splink default":
            return g.loc[g.method == SPLINK_DEFAULT, f"true_{query}"].iloc[0]
        if chooser == "hard veto":
            return g.loc[g.method == "hybrid-any-0.2+hardveto", f"true_{query}"].iloc[0]

    lines, share = [], []
    for pool in ("base", "ext"):
        for ch, q in [("oracle", None)] + [("VetoQ-E", q) for q in QS]:
            vals = {}
            for query in QUERIES:
                vals[query] = np.mean([pick(g if pool == "ext" else g[g.pool == "base"], ch, q, query)
                                       for _, g in inst])
            vals["mean"] = np.mean([vals[x] for x in QUERIES])
            lines.append(pd.Series(vals, name=f"{pool}: {ch}" + ("" if q is None else f"(q={q})")))
    for ch in ("Splink default", "hard veto"):
        vals = {query: np.mean([pick(g, ch, None, query) for _, g in inst]) for query in QUERIES}
        vals["mean"] = np.mean([vals[x] for x in QUERIES])
        lines.append(pd.Series(vals, name=ch))
    print(f"\n=== E-E with external candidates: true error of the chosen equivalence, mean over {inst.ngroups} "
          f"Hard-multi instances ===")
    print(pd.DataFrame(lines).to_string(float_format=lambda v: f"{v:.3f}"))
    for q in QS:
        picks = []
        for _, g in inst:
            for query in QUERIES:
                picks.append(g.loc[C.choose(g, f"est{q}_{query}", f"estM_{q}"), "pool"] == "ext")
        share.append(f"q={q}: {np.mean(picks):.2f}")
    print("\nshare of (instance, query) choices of VetoQ-E on the ext pool that pick a Splink candidate: "
          + ", ".join(share))
    print("\nper query, instances where a Splink candidate is the oracle choice (ext pool):")
    for query in QUERIES:
        cnt = sum(g.loc[g[f"true_{query}"].idxmin(), "pool"] == "ext" for _, g in inst)
        print(f"  {query}: {cnt} of {inst.ngroups}")


if __name__ == "__main__":
    main()
