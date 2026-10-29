"""Equivalence construction for T1: exact / normalized / similarity grouping with optional Context Veto.

group(df, backend, eps, linkage, veto) -> np.ndarray of class labels, one per row of df.
  backend : "exact" | "norm" | "edit" | "emb" | "hybrid"
  linkage : "any"  (connected components; SGB-Any, single linkage)
            "all"  (complete linkage at distance eps; SGB-All-like clique/diameter bound)
  veto    : "none" | "hard" (Context Veto, min with context sim) | "soft" (context sim >= THETA counts as 1)
Similarities are computed once per distinct (key, context) combination, as in hard_bench.canon.
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage as hlinkage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import squareform

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hard_bench import norm, sim_matrix          # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
THETA = 0.5          # soft veto threshold (E17)
GEO_KM = 25.0        # geo context: two points within this distance agree


_SIM_CACHE = {}


def _sim(vals, backend, model):
    """sim_matrix with a cache keyed by (backend, values), since the method grid reuses the same values."""
    key = (backend, model, tuple(vals))
    if key not in _SIM_CACHE:
        if backend == "hybrid" and ("edit", model, tuple(vals)) in _SIM_CACHE and ("emb", model, tuple(vals)) in _SIM_CACHE:
            _SIM_CACHE[key] = np.maximum(_SIM_CACHE[("edit", model, tuple(vals))], _SIM_CACHE[("emb", model, tuple(vals))])
        else:
            _SIM_CACHE[key] = sim_matrix(vals, backend, model)
    return _SIM_CACHE[key]


def clear_cache():
    _SIM_CACHE.clear()


def _geo_sim(lat, lon):
    lat, lon = np.radians(lat), np.radians(lon)
    dlat = lat[:, None] - lat[None, :]
    dlon = lon[:, None] - lon[None, :]
    a = np.sin(dlat / 2) ** 2 + np.cos(lat[:, None]) * np.cos(lat[None, :]) * np.sin(dlon / 2) ** 2
    d = 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    s = (d <= GEO_KM).astype(float)
    missing = np.isnan(lat)
    s[missing, :] = 1.0                              # unknown context cannot veto
    s[:, missing] = 1.0
    return s


def _cluster(S, eps, linkage):
    if linkage == "any":
        return connected_components(csr_matrix(S >= 1 - eps - 1e-12), directed=False)[1]
    if len(S) == 1:
        return np.zeros(1, dtype=int)
    D = np.clip(1 - S, 0, None)
    np.fill_diagonal(D, 0)
    D = np.maximum(D, D.T)
    return fcluster(hlinkage(squareform(D, checks=False), "complete"), t=eps + 1e-12, criterion="distance")


def group(df, backend, eps=0.2, linkage="any", veto="none", ctx_kind=None, model=MODEL):
    keys = df["K"].astype(str)
    if backend == "exact":
        return keys.to_numpy()
    if backend == "norm":
        return keys.map(norm).to_numpy()
    use_ctx = veto != "none" and ctx_kind is not None
    keys = [str(k) for k in keys]
    if use_ctx and ctx_kind == "geo":
        ctx = [f"{float(a):.4f}|{float(b):.4f}" for a, b in zip(df["LAT"], df["LON"])]
    elif use_ctx:
        ctx = [str(c) for c in df["CTX"]]
    else:
        ctx = [""] * len(df)
    combos = list(dict.fromkeys(zip(keys, ctx)))
    vals = list(dict.fromkeys(k for k, _ in combos))
    vi = {v: i for i, v in enumerate(vals)}
    a = np.array([vi[k] for k, _ in combos])
    Sv = _sim(vals, backend, model)
    S = Sv[np.ix_(a, a)]
    if use_ctx:
        cvals = list(dict.fromkeys(c for _, c in combos))
        ci = {c: i for i, c in enumerate(cvals)}
        b = np.array([ci[c] for _, c in combos])
        if ctx_kind == "geo":
            lat = np.array([float(c.split("|")[0]) for c in cvals])
            lon = np.array([float(c.split("|")[1]) for c in cvals])
            Sc = _geo_sim(lat, lon)
        else:
            Sc = _sim(cvals, backend, model)
        if veto == "soft":
            Sc = np.where(Sc >= THETA, 1.0, Sc)
        S = np.minimum(S, Sc[np.ix_(b, b)])
    lab = _cluster(S, eps, linkage)
    m = dict(zip(combos, lab))
    return np.array([m[k] for k in zip(keys, ctx)])


def method_grid(ctx_kind):
    """The candidate equivalences used in E-A (name -> kwargs)."""
    g = {"exact": dict(backend="exact"), "norm": dict(backend="norm")}
    for be in ("edit", "emb", "hybrid"):
        for link in ("any", "all"):
            for eps in (0.1, 0.2, 0.3):
                g[f"{be}-{link}-{eps}"] = dict(backend=be, linkage=link, eps=eps)
    if ctx_kind is not None:
        for v in ("hard", "soft"):
            for link in ("any", "all"):
                g[f"hybrid-{link}-0.2+{v}veto"] = dict(backend="hybrid", linkage=link, eps=0.2, veto=v)
    return g
