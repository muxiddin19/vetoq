"""E-E on real data: choosing the equivalence per query with a noisy, partially observed witness (Hospital).

Key: HospitalName (dirty). Entity: ProviderNumber (clean). Context: State.
Witness: the *dirty* PhoneNumber, observed on a random fraction q of rows; it stands in for the entity when
estimating query errors, exactly as in run_ed.py, and is itself noisy (typos in the dirty table).
Queries: Q1 COUNT(*) per hospital, Q2 COUNT(DISTINCT hospital), Q3 equi-join of two random halves
(error = (false + missed pairs) / true pairs).

Usage: python -P exp_t1/run_ee_real.py
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import contingency as C          # noqa: E402
import grouping as G             # noqa: E402
import exp2_data as E            # noqa: E402

QS = [0.1, 0.3, 1.0]
QUERIES = ["Q1", "Q2", "Q3"]


def errors(lab, truth, mask, side):
    lab, truth, side = lab[mask], truth[mask], side[mask]
    n = C.table(lab, truth)
    ms = C.masses(n)
    q1 = float((ms["m"] - ms["s"]).abs().sum() / ms["size"].sum())
    q2 = abs(n.shape[0] - n.shape[1]) / n.shape[1]
    j = C.formula_join(C.table(lab[side], truth[side]), C.table(lab[~side], truth[~side]))
    q3 = (j["FP"] + j["FN"]) / max(j["TP"] + j["FN"], 1)
    return {"Q1": q1, "Q2": q2, "Q3": q3}


def main():
    d, c = E.load("hospital")
    df = pd.DataFrame({"K": d["HospitalName"], "E": c["ProviderNumber"], "CTX": d["State"]})
    truth = df["E"].astype(str).to_numpy()
    witness = d["PhoneNumber"].astype(str).to_numpy()
    print(f"witness noise: {np.mean(d['PhoneNumber'].to_numpy() != c['PhoneNumber'].to_numpy()):.3f} of phone cells dirty")
    rows, wrows = [], []
    G.clear_cache()
    cands = {name: np.asarray(G.group(df, ctx_kind="str", **kw)).astype(str)
             for name, kw in G.method_grid("str").items()}
    full = np.ones(len(df), dtype=bool)
    for rep in range(20):                                     # resample witness masks and join halves
        rng = np.random.default_rng(rep)
        side = rng.random(len(df)) < 0.5
        draw = rng.random(len(df))
        for name, lab in cands.items():
            rec = dict(rep=rep, method=name, **{f"true_{k}": v for k, v in errors(lab, truth, full, side).items()})
            for q in QS:
                m = draw < q
                rec.update({f"est{q}_{k}": v for k, v in errors(lab, witness, m, side).items()})
                rec[f"estF1_{q}"] = C.er_metrics(C.table(lab[m], witness[m]))["pairF1"]
                rec[f"estM_{q}"] = float(C.masses(C.table(lab[m], witness[m]))["m"].sum() / max(m.sum(), 1))
            rows.append(rec)
        for q in QS:                                          # Witness-GB: group by the observed phone itself
            m = draw < q
            key = df["K"].astype(str).map(G.norm).to_numpy()
            maj = pd.DataFrame({"k": key[m], "w": witness[m]}).groupby("k", sort=False)["w"].agg(
                lambda s_: s_.value_counts().index[0]).to_dict()
            wl = np.array(["w:" + witness[i] if m[i] else ("w:" + maj[key[i]] if key[i] in maj else "k:" + key[i])
                           for i in range(len(df))])
            wrows.append(dict(rep=rep, method=f"Witness-GB(q={q})",
                              **{f"true_{k}": v for k, v in errors(wl, truth, full, side).items()}))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(os.path.dirname(HERE), "gpu_out", "t1_ee_hospital.csv"), index=False)
    tab = res.groupby("method")[[f"true_{q}" for q in QUERIES]].mean()
    tab["mean"] = tab.mean(axis=1)
    print("\n=== Hospital: true error per candidate (mean over 20 join splits) ===")
    print(tab.sort_values("mean").head(10).to_string(float_format=lambda v: f"{v:.3f}"))
    best_single = tab["mean"].idxmin()
    lines = []
    g_all = list(res.groupby("rep"))

    def chosen(ch, q, query):
        out = []
        for _, g in g_all:
            if ch == "oracle":
                out.append(g[f"true_{query}"].min())
            elif ch == "best-single":
                out.append(g.loc[g.method == best_single, f"true_{query}"].iloc[0])
            elif ch in ("exact", "norm", "hybrid-any-0.2", "hybrid-any-0.2+hardveto"):
                out.append(g.loc[g.method == ch, f"true_{query}"].iloc[0])
            elif ch == "VetoQ-E":
                out.append(g.loc[C.choose(g, f"est{q}_{query}", f"estM_{q}"), f"true_{query}"])
            elif ch == "F1-E":
                out.append(g.loc[C.choose(g, f"estF1_{q}", f"estM_{q}", maximize=True), f"true_{query}"])
        return float(np.mean(out))

    for ch, q in [("oracle", None), ("best-single", None), ("exact", None), ("norm", None),
                  ("hybrid-any-0.2", None), ("hybrid-any-0.2+hardveto", None)] + \
                 [(c_, q_) for q_ in QS for c_ in ("VetoQ-E", "F1-E")]:
        vals = {query: chosen(ch, q, query) for query in QUERIES}
        vals["mean"] = np.mean(list(vals.values()))
        lines.append(pd.Series(vals, name=ch if q is None else f"{ch}(q={q})"))
    wres = pd.DataFrame(wrows)
    wres.to_csv(os.path.join(os.path.dirname(HERE), "gpu_out", "t1_ee_hospital_witness.csv"), index=False)
    for name, g in wres.groupby("method", sort=False):
        vals = g[[f"true_{x}" for x in QUERIES]].mean()
        vals.index = QUERIES
        vals["mean"] = vals.mean()
        lines.append(pd.Series(vals, name=name))
    print(f"\n=== Hospital E-E: true error of the chosen candidate (best-single = {best_single}) ===")
    print(pd.DataFrame(lines).to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
