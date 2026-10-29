"""T1 operator error calculus (paper §III, Lemmas 1-4 and 6; proofs in TECHNICAL_REPORT.md) and ER quality metrics.

Two independent paths per operator:
  formula_*  -- computed from the contingency table n[c, e] only
  execute_*  -- the SQL-style operator actually run with pandas on labels
E-C checks that the two agree exactly; E-A compares ER metrics with operator errors.
"""
import math
from collections import Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- contingency
def table(pred, truth):
    """n[c, e] as a DataFrame (rows: classes c, cols: entities e).

    n.attrs["first"] maps (c, e) to the position of the first tuple of e in c (used only to break ties)."""
    p, t = pd.Series(np.asarray(pred), name="c"), pd.Series(np.asarray(truth), name="e")
    n = pd.crosstab(p, t)
    pos = pd.DataFrame({"c": p, "e": t, "i": np.arange(len(p))})
    n.attrs["first"] = pos.groupby(["c", "e"], sort=False)["i"].min().to_dict()
    return n


def rep_class(n):
    """c*(e) = argmax_c n[c,e]; ties -> the smallest (purest) class, then the class holding e's first tuple.

    The rule depends on the partition and the tuple order only, never on class labels (paper §II)."""
    vals = n.to_numpy()
    top = vals == vals.max(axis=0, keepdims=True)
    rep = n.idxmax(axis=0)
    tied = np.flatnonzero(top.sum(axis=0) > 1)
    if len(tied):
        size = vals.sum(axis=1)
        first = n.attrs.get("first", {})
        for j in tied:
            rows = np.flatnonzero(top[:, j])
            rows = rows[size[rows] == size[rows].min()]
            e = n.columns[j]
            rep.iloc[j] = min((n.index[i] for i in rows), key=lambda c: first.get((c, e), np.inf))
    return rep


# ---------------------------------------------------------------- choosers (E-E)
def choose(g, est_col, tie_col, maximize=False, tol=1e-12):
    """Row label chosen from candidate rows g: best estimate in est_col; ties within tol go to the smallest
    tie_col (the estimated merge mass, since merges are the error closure amplifies), then to grid order."""
    v = g[est_col].to_numpy(dtype=float)
    v = -v if maximize else v
    best = np.nanmin(v)
    ties = g.index[np.abs(v - best) <= tol]
    if len(ties) == 1:
        return ties[0]
    return g.loc[ties, tie_col].idxmin()


def tie_range(g, est_col, true_col, maximize=False, tol=1e-12):
    """(min, max) true error over all candidates tied for the best estimate."""
    v = g[est_col].to_numpy(dtype=float)
    v = -v if maximize else v
    t = g.loc[g.index[np.abs(v - np.nanmin(v)) <= tol], true_col]
    return float(t.min()), float(t.max())


def masses(n):
    """Per-entity split mass s(e) and merge mass m(e)."""
    rep = rep_class(n)
    size_e = n.sum(axis=0)
    size_c = n.sum(axis=1)
    core = pd.Series({e: n.at[rep[e], e] for e in n.columns})
    s = size_e - core
    m = size_c.loc[rep.values].values - core
    return pd.DataFrame({"rep": rep, "size": size_e, "core": core, "s": s, "m": pd.Series(m, index=n.columns)})


# ---------------------------------------------------------------- Lemma 1/2: GROUP BY per entity
def formula_groupby_sum(n_w, ms):
    """Lemma 2 with weights: n_w is the contingency table of SUM(A) instead of counts."""
    rep = ms["rep"]
    out = {}
    for e in n_w.columns:
        intruders = n_w.loc[rep[e]].sum() - n_w.at[rep[e], e]
        missing = n_w[e].sum() - n_w.at[rep[e], e]
        out[e] = intruders - missing
    return pd.Series(out)


def execute_groupby(pred, truth, weight=None):
    """Run GROUP BY pred, aggregate, and read each entity's answer from its plurality group."""
    df = pd.DataFrame({"c": np.asarray(pred), "e": np.asarray(truth),
                       "w": 1.0 if weight is None else np.asarray(weight, dtype=float)})
    agg = df.groupby("c")["w"].sum()                       # the query result
    true = df.groupby("e")["w"].sum()
    rep = rep_class(table(pred, truth))
    return pd.Series({e: agg[rep[e]] - true[e] for e in true.index})


# ---------------------------------------------------------------- Lemma 3: COUNT(DISTINCT)
def formula_count_distinct(n):
    """Sum over components of the class-entity bipartite graph of (p_i - q_i)."""
    nz = n.values > 0
    classes, ents = list(n.index), list(n.columns)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, j in zip(*np.nonzero(nz)):
        a, b = find(("c", classes[i])), find(("e", ents[j]))
        if a != b:
            parent[a] = b
    comp = Counter()
    for c in classes:
        comp[find(("c", c))] += 1
    for e in ents:
        comp[find(("e", e))] -= 1
    return int(sum(comp.values())), comp


def execute_count_distinct(pred, truth):
    return int(pd.Series(pred).nunique() - pd.Series(truth).nunique())


# ---------------------------------------------------------------- Lemma 4: equi-join
def formula_join(nR, nS):
    cls = nR.index.union(nS.index)
    ent = nR.columns.union(nS.columns)
    R = nR.reindex(index=cls, columns=ent, fill_value=0).values.astype(np.int64)
    S = nS.reindex(index=cls, columns=ent, fill_value=0).values.astype(np.int64)
    tp = int((R * S).sum())
    same_class = int((R.sum(1) * S.sum(1)).sum())          # all pairs with kappa(r)=kappa(s)
    same_entity = int((R.sum(0) * S.sum(0)).sum())         # all pairs with e(r)=e(s)
    return {"TP": tp, "FP": same_class - tp, "FN": same_entity - tp}


def execute_join(predR, truthR, predS, truthS):
    R = pd.DataFrame({"c": predR, "e": truthR})
    S = pd.DataFrame({"c": predS, "e": truthS})
    j = R.merge(S, on="c", suffixes=("_r", "_s"))           # the semantic join
    tp = int((j["e_r"] == j["e_s"]).sum())
    fp = int((j["e_r"] != j["e_s"]).sum())
    jt = R.merge(S, on="e", suffixes=("_r", "_s"))          # the ground-truth join
    fn = int((jt["c_r"] != jt["c_s"]).sum())
    return {"TP": tp, "FP": fp, "FN": fn}


# ---------------------------------------------------------------- Lemma 6: selection
def execute_select(pred, truth, query_rows):
    """query_rows: {entity: index of a tuple holding the query value}; returns FP/FN per entity."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    out = {}
    for e, i in query_rows.items():
        hit = pred == pred[i]
        out[e] = (int((hit & (truth != e)).sum()), int((~hit & (truth == e)).sum()))
    return out


def formula_select(n, pred, query_rows):
    pred = np.asarray(pred)
    out = {}
    for e, i in query_rows.items():
        cq = pred[i]
        out[e] = (int(n.loc[cq].sum() - n.at[cq, e]), int(n[e].sum() - n.at[cq, e]))
    return out


# ---------------------------------------------------------------- ER quality metrics
def _pairs(x):
    x = np.asarray(x, dtype=np.int64)
    return int((x * (x - 1) // 2).sum())


def er_metrics(n):
    v = n.values.astype(np.int64)
    N = v.sum()
    tp = _pairs(v)
    pp, tt = _pairs(v.sum(1)), _pairs(v.sum(0))
    p = tp / pp if pp else 1.0
    r = tp / tt if tt else 1.0
    pf1 = 2 * p * r / (p + r) if p + r else 0.0
    # B-cubed: per tuple, precision = n_ce/|c|, recall = n_ce/|e|
    cs, es = v.sum(1, keepdims=True), v.sum(0, keepdims=True)
    bp = float((v * v / np.where(cs == 0, 1, cs)).sum() / N)
    br = float((v * v / np.where(es == 0, 1, es)).sum() / N)
    bf1 = 2 * bp * br / (bp + br) if bp + br else 0.0
    # Variation of information, natural log
    pj = v / N
    pc, pe = pj.sum(1), pj.sum(0)
    nz = pj > 0
    vi = float(-(pj[nz] * (np.log(pj[nz] / pc[:, None].repeat(v.shape[1], 1)[nz])
                           + np.log(pj[nz] / pe[None, :].repeat(v.shape[0], 0)[nz]))).sum())
    # Basic merge distance (GMD with unit costs): splits + merges on the bare-necessities path
    k_c = (v > 0).sum(1)
    k_e = (v > 0).sum(0)
    bmd = int((k_c - 1).clip(min=0).sum() + (k_e - 1).clip(min=0).sum())
    return {"pairF1": pf1, "B3F1": bf1, "VI": vi, "BMD": bmd}


def operator_errors(n):
    """Headline per-table operator errors, all from the contingency table."""
    ms = masses(n)
    cnt_err = ((ms["m"] - ms["s"]).abs() / ms["size"]).mean()
    cd, _ = formula_count_distinct(n)
    v = n.to_numpy()
    mixed = int(v[(v > 0).sum(1) > 1].sum())                # tuples in classes meeting >= 2 entities
    return {"COUNT_err": float(cnt_err), "CD_err": abs(cd) / n.shape[1],
            "M": int(ms["m"].sum()), "S": int(ms["s"].sum()), "X": mixed}
