"""E-D (pipeline queries) and E-E (choosing the equivalence per query) on the Hard-2000 multi-table data.

Queries over orders(country dirty), customers(country dirty), countries(clean dimension):
  Q1  SELECT country, SUM(amount) FROM orders GROUP BY country             -- per-entity aggregate
  Q2  SELECT COUNT(DISTINCT country) FROM orders                           -- entity count
  Q3  orders JOIN customers ON country ~ country, pairs per country         -- join then aggregate
  Q4  orders JOIN countries ON country ~ name, SUM(amount) BY continent     -- lookup to a clean dimension
One equivalence kappa is computed on the union of all country values (orders, customers, dimension), so all
queries see the same classes. Error of a query = relative L1 error of its answer vector against the truth.

Choosers (E-E):
  oracle        per-query best candidate in hindsight (lower bound)
  best-single   the one candidate with the lowest mean error over all queries and instances (hindsight)
  default-*     exact / norm / hybrid-any-0.2
  VetoQ-R       rule from theorem C3: hybrid + hard Context Veto for every query
  VetoQ-E(q)    per query, the candidate with the lowest error *estimated* on rows whose calling code
                (witness) is observed, with the code standing in for the entity; q = observed fraction
  F1-E(q)       one candidate for all queries, maximizing pairwise F1 estimated on the same witnessed rows
  Ties in an estimate go to the candidate with the smallest estimated merge mass (contingency.choose); the
  report also gives the range of true errors over the tied candidates.
  Witness-GB(q) not a candidate: group by the observed witness itself, propagating the majority witness to
                unwitnessed rows with the same normalized key (a direct use of the identifier)
  --wit-noise p replaces each recorded witness by another entity's code with probability p (default 0)

Usage: python -P exp_t1/run_ed.py [--seeds 0 1 2 3 4] [--wit-noise 0.1 --tag _witnoise0.1]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import contingency as C          # noqa: E402
import grouping as G             # noqa: E402
from hard_multi import generate  # noqa: E402

QUERIES = ["Q1", "Q2", "Q3", "Q4"]
QS = [0.1, 0.3, 1.0]


def union_frame(countries, customers, orders):
    o = pd.DataFrame({"tbl": "O", "K": orders["country"], "CTX": orders["capital"], "E": orders["E"],
                      "code": orders["code"], "amount": orders["amount"]})
    c = pd.DataFrame({"tbl": "C", "K": customers["country"], "CTX": customers["capital"], "E": customers["E"],
                      "code": customers["code"], "amount": 0.0})
    d = pd.DataFrame({"tbl": "D", "K": countries["name"], "CTX": countries["capital"], "E": countries["name"],
                      "code": countries["code"], "amount": 0.0, "cont": countries["continent"]})
    u = pd.concat([o, c, d], ignore_index=True)
    return u


def rel_l1(ans, true):
    idx = true.index.union(ans.index)
    a, t = ans.reindex(idx, fill_value=0.0), true.reindex(idx, fill_value=0.0)
    return float((a - t).abs().sum() / max(t.abs().sum(), 1e-12))


def query_errors(u, lab, truth, mask, cont_of):
    """Errors of Q1-Q4 with `truth` as the entity id, using only rows in `mask` (dimension rows always kept)."""
    keep = mask | (u["tbl"] == "D").to_numpy()
    U = u[keep].assign(c=lab[keep], t=truth[keep])
    O, Cu, Dm = U[U.tbl == "O"], U[U.tbl == "C"], U[U.tbl == "D"]
    out = {}
    if len(O) == 0:
        return {q: np.nan for q in QUERIES}
    nO = C.table(O["c"].to_numpy(), O["t"].to_numpy())
    rep = C.rep_class(nO)
    # Q1: per-entity SUM read from the plurality class
    sum_c = O.groupby("c")["amount"].sum()
    true1 = O.groupby("t")["amount"].sum()
    out["Q1"] = rel_l1(pd.Series({e: sum_c[rep[e]] for e in true1.index}), true1)
    # Q2: number of distinct entities among orders
    out["Q2"] = abs(O["c"].nunique() - O["t"].nunique()) / O["t"].nunique()
    # Q3: join orders with customers on the class, pairs per entity (read from the order's plurality class)
    co, cc = O.groupby("c").size(), Cu.groupby("c").size()
    to, tc = O.groupby("t").size(), Cu.groupby("t").size()
    ans3 = pd.Series({e: co.get(rep[e], 0) * cc.get(rep[e], 0) for e in to.index})
    true3 = pd.Series({e: to.get(e, 0) * tc.get(e, 0) for e in to.index})
    out["Q3"] = rel_l1(ans3, true3)
    # Q4: semantic lookup into the clean dimension, then SUM(amount) by continent
    dims = Dm.groupby("c")["t"].apply(list)
    ans4 = {}
    for c, amt in O.groupby("c")["amount"].sum().items():
        for d in dims.get(c, []):                         # 0 matches: row lost; >1: duplicated (join semantics)
            k = cont_of[d]
            ans4[k] = ans4.get(k, 0.0) + amt
    true4 = O.assign(k=O["t"].map(cont_of)).groupby("k")["amount"].sum()
    out["Q4"] = rel_l1(pd.Series(ans4, dtype=float), true4)
    return out


def pair_f1(lab, truth, mask):
    n = C.table(lab[mask], truth[mask])
    return C.er_metrics(n)["pairF1"]


def merge_frac(lab, truth, mask):
    """Estimated merge mass sum_e m(e)/n on the witnessed rows (the chooser's tie-break)."""
    return float(C.masses(C.table(lab[mask], truth[mask]))["m"].sum() / max(mask.sum(), 1))


def witness_labels(u, code, mask):
    """Witness-GB: class = observed witness; unwitnessed rows take the majority witness of their normalized key."""
    key = u["K"].astype(str).map(G.norm).to_numpy()
    dim = (u["tbl"] == "D").to_numpy()
    obs = pd.DataFrame({"k": key[mask], "w": code[mask]})
    maj = obs.groupby("k", sort=False)["w"].agg(lambda s: s.value_counts().index[0]).to_dict()
    out = []
    for i in range(len(u)):
        if dim[i] or mask[i]:
            out.append("w:" + code[i])
        elif key[i] in maj:
            out.append("w:" + maj[key[i]])
        else:
            out.append("k:" + key[i])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--wit-noise", type=float, default=0.0)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    rows, wrows = [], []
    for seed in args.seeds:
        for rate in (0.1, 0.2, 0.3):
            countries, customers, orders = generate(seed, rate, wit_noise=args.wit_noise)
            u = union_frame(countries, customers, orders)
            truth = u["E"].astype(str).to_numpy()
            code = u["code"].astype(str).to_numpy()
            cont_true = dict(zip(countries["name"], countries["continent"]))
            cont_code = dict(zip(countries["code"].astype(str), countries["continent"]))
            rng = np.random.default_rng(seed * 100 + int(rate * 10))
            u_draw = rng.random(len(u))
            masks = {q: (u_draw < q) & (u["tbl"] != "D").to_numpy() for q in QS}
            full = (u["tbl"] != "D").to_numpy()
            G.clear_cache()
            for name, kw in G.method_grid("str").items():
                lab = np.asarray(G.group(u, ctx_kind="str", **kw)).astype(str)
                true_err = query_errors(u, lab, truth, full, cont_true)
                rec = dict(seed=seed, rate=rate, method=name, **{f"true_{q}": v for q, v in true_err.items()})
                for q in QS:
                    est = query_errors(u, lab, code, masks[q], cont_code)
                    rec.update({f"est{q}_{k}": v for k, v in est.items()})
                    rec[f"estF1_{q}"] = pair_f1(lab, code, masks[q])
                    rec[f"estM_{q}"] = merge_frac(lab, code, masks[q])
                rows.append(rec)
            for q in QS:
                wl = witness_labels(u, code, masks[q])
                wrows.append(dict(seed=seed, rate=rate, method=f"Witness-GB(q={q})",
                                  **{f"true_{k}": v for k, v in query_errors(u, wl, truth, full, cont_true).items()}))
            print(f"[seed {seed} rate {rate}] done", flush=True)
    res = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(HERE), "gpu_out", f"t1_ed{args.tag}.csv")
    res.to_csv(out, index=False)
    wres = pd.DataFrame(wrows)
    wres.to_csv(os.path.join(os.path.dirname(HERE), "gpu_out", f"t1_ed{args.tag}_witness.csv"), index=False)
    print(f"witness noise: {args.wit_noise}")
    report(res, wres)


def report(res, wres=None):
    inst = res.groupby(["seed", "rate"])
    tq = [f"true_{q}" for q in QUERIES]
    print("\n=== E-D: true error of each candidate, mean over instances (lower is better) ===")
    tab = res.groupby("method")[tq].mean()
    tab["mean"] = tab.mean(axis=1)
    print(tab.sort_values("mean").to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n=== best candidate per query (instances where it is best) ===")
    for q in QUERIES:
        best = res.loc[res.groupby(["seed", "rate"])[f"true_{q}"].idxmin(), "method"].value_counts().head(4)
        print(f"{q}: " + ", ".join(f"{m} ({c})" for m, c in best.items()))
    best_single = tab["mean"].idxmin()

    def pick(g, chooser, q, query):
        if chooser == "oracle":
            return g[f"true_{query}"].min()
        if chooser == "best-single":
            return g.loc[g.method == best_single, f"true_{query}"].iloc[0]
        if chooser.startswith("default-") or chooser == "VetoQ-R":
            m = {"default-exact": "exact", "default-norm": "norm", "default-hybrid": "hybrid-any-0.2",
                 "VetoQ-R": "hybrid-any-0.2+hardveto"}[chooser]
            return g.loc[g.method == m, f"true_{query}"].iloc[0]
        if chooser == "VetoQ-E":
            return g.loc[C.choose(g, f"est{q}_{query}", f"estM_{q}"), f"true_{query}"]
        if chooser == "F1-E":
            return g.loc[C.choose(g, f"estF1_{q}", f"estM_{q}", maximize=True), f"true_{query}"]

    print(f"\n=== E-E: true error of the chosen candidate, mean over {inst.ngroups} instances "
          f"(best-single = {best_single}) ===")
    choosers = [("oracle", None), ("best-single", None), ("default-exact", None), ("default-norm", None),
                ("default-hybrid", None), ("VetoQ-R", None)] + \
               [(c, q) for q in QS for c in ("VetoQ-E", "F1-E")]
    lines = []
    for ch, q in choosers:
        vals = {query: np.mean([pick(g, ch, q, query) for _, g in inst]) for query in QUERIES}
        vals["mean"] = np.mean(list(vals.values()))
        lines.append(pd.Series(vals, name=ch if q is None else f"{ch}(q={q})"))
    if wres is not None and len(wres):
        for name, g in wres.groupby("method", sort=False):
            vals = g[[f"true_{x}" for x in QUERIES]].mean()
            vals.index = QUERIES
            vals["mean"] = vals.mean()
            lines.append(pd.Series(vals, name=name))
    print(pd.DataFrame(lines).to_string(float_format=lambda v: f"{v:.3f}"))
    print("\n=== VetoQ-E: range of the true error over candidates tied for the best estimate (mean over instances) ===")
    for q in QS:
        lo, hi, nt = {}, {}, 0
        for query in QUERIES:
            rr = [C.tie_range(g, f"est{q}_{query}", f"true_{query}") for _, g in inst]
            lo[query], hi[query] = np.mean([a for a, _ in rr]), np.mean([b for _, b in rr])
            nt += sum(a != b for a, b in rr)
        print(f"q={q}: lowest {np.mean(list(lo.values())):.3f}, highest {np.mean(list(hi.values())):.3f} "
              f"(mean over queries); decisions with ties that matter: {nt} of {inst.ngroups * len(QUERIES)}")


if __name__ == "__main__":
    main()
