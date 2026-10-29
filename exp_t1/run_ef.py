"""E-F: cost and scale. Scalable grouping with ANN candidates on MusicBrainz 20K and 200K (all records).

Nodes = distinct (normalized title, normalized artist) combinations. Candidate value pairs come from exact
top-k cosine neighbours of the title under two representations (MiniLM sentence embedding; character 2-4-gram
TF-IDF reduced to 128 dimensions by truncated SVD), computed in blocks. Each candidate is scored with the
hybrid similarity max(edit ratio, embedding cosine) and kept if >= 1 - eps; with the Context Veto the artist
pair must also be >= 1 - eps under the same hybrid similarity. Classes are connected components.

Reported: wall-clock per stage, operator errors (COUNT per entity, COUNT DISTINCT, join false/missed pairs)
and pairwise F1, for exact, normalized, hybrid and hybrid + Context Veto.

Usage: python -P exp_t1/run_ef.py [--k 10] [--eps 0.2]
"""
import argparse
import os
import sys
import time
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import contingency as C                       # noqa: E402
from hard_bench import norm                   # noqa: E402
from embedding_backend import embed           # noqa: E402

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DATA = os.path.join(os.path.dirname(HERE), "data_t1")


def topk(X, k, block=512):
    """Exact top-k cosine neighbours (rows of X are L2-normalized), excluding self."""
    n = X.shape[0]
    out = np.empty((n, k), dtype=np.int64)
    for s in range(0, n, block):
        sims = X[s:s + block] @ X.T
        for i in range(sims.shape[0]):
            sims[i, s + i] = -1.0
        idx = np.argpartition(-sims, kth=min(k, n - 1) - 1, axis=1)[:, :k]
        out[s:s + block] = idx
    return out


def hybrid_pairs(texts, emb, pairs):
    cos = np.einsum("ij,ij->i", emb[pairs[:, 0]], emb[pairs[:, 1]])
    ed = np.array([SequenceMatcher(None, texts[a], texts[b]).ratio() for a, b in pairs])
    return np.maximum(np.clip(cos, 0, 1), ed)


def run(path, k, eps, label):
    t = {}
    t0 = time.perf_counter()
    d = pd.read_csv(path, dtype=str, keep_default_na=False, usecols=["CID", "title", "artist"])
    t["load"] = time.perf_counter() - t0
    truth = d["CID"].to_numpy()
    titles = d["title"].map(norm).to_numpy()
    artists = d["artist"].map(norm).to_numpy()
    rows = []
    for name, lab in (("exact", d["title"].to_numpy()), ("norm", titles)):
        rows.append(evaluate(name, lab, truth, 0.0, label))

    t0 = time.perf_counter()
    combo = pd.Series(list(zip(titles, artists)))
    codes, uniq = pd.factorize(combo)
    ut = np.array([u[0] for u in uniq], dtype=object)
    ua = np.array([u[1] for u in uniq], dtype=object)
    tvals, tidx = np.unique(ut, return_inverse=True)
    avals, aidx = np.unique(ua, return_inverse=True)
    t["factorize"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    E_t = np.asarray(embed(list(tvals), model_name=MODEL), dtype=np.float32)
    E_a = np.asarray(embed(list(avals), model_name=MODEL), dtype=np.float32)
    t["embed"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    tf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1).fit_transform(tvals)
    svd = TruncatedSVD(128, random_state=0).fit_transform(tf).astype(np.float32)
    svd /= np.maximum(np.linalg.norm(svd, axis=1, keepdims=True), 1e-12)
    t["tfidf"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    nb = np.concatenate([topk(E_t, k), topk(svd, k)], axis=1)
    src = np.repeat(np.arange(len(tvals)), nb.shape[1])
    pairs = np.stack([src, nb.ravel()], axis=1)
    pairs = np.unique(np.sort(pairs, axis=1), axis=0)
    pairs = pairs[pairs[:, 0] != pairs[:, 1]]
    t["knn"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_title = hybrid_pairs(tvals, E_t, pairs)
    keep = pairs[s_title >= 1 - eps]
    t["score"] = time.perf_counter() - t0
    # combos sharing a title value are joined directly; title edges join all combos of the two titles
    for veto in (False, True):
        t0 = time.perf_counter()
        n_c = len(uniq)
        by_title = pd.Series(np.arange(n_c)).groupby(tidx).apply(list)
        edges = []
        for lst in by_title:                                         # same title, different artist
            for a in lst[1:]:
                edges.append((lst[0], a))
        e_same = np.array(edges, dtype=np.int64).reshape(-1, 2)
        cand = []
        tl = by_title.to_dict()
        for a, b in keep:
            for x in tl[a]:
                for y in tl[b]:
                    cand.append((x, y))
        cand = np.array(cand, dtype=np.int64).reshape(-1, 2)
        all_e = np.concatenate([e_same, cand]) if len(cand) else e_same
        if veto and len(all_e):
            ap = np.stack([aidx[all_e[:, 0]], aidx[all_e[:, 1]]], axis=1)
            same = ap[:, 0] == ap[:, 1]
            sa = np.ones(len(ap))
            if (~same).any():
                sa[~same] = hybrid_pairs(avals, E_a, ap[~same])
            all_e = all_e[sa >= 1 - eps]
        g = coo_matrix((np.ones(len(all_e)), (all_e[:, 0], all_e[:, 1])), shape=(n_c, n_c))
        comp = connected_components(g, directed=False)[1]
        lab = comp[codes]
        dt = time.perf_counter() - t0
        name = "hybrid-any+hardveto" if veto else "hybrid-any"
        base = t["factorize"] + t["embed"] + t["tfidf"] + t["knn"] + t["score"]
        rows.append(evaluate(name, lab.astype(str), truth, base + dt, label))
    tim = pd.Series(t)
    return rows, tim, dict(rows=len(d), entities=len(np.unique(truth)), distinct_titles=len(tvals),
                           distinct_artists=len(avals), combos=len(uniq), cand_pairs=len(pairs),
                           kept_pairs=len(keep))


def _c2(x):
    x = np.asarray(x, dtype=np.int64)
    return int((x * (x - 1) // 2).sum())


def evaluate(name, lab, truth, sec, label):
    """Same quantities as contingency.py, computed on the sparse contingency table (large instances)."""
    lab = np.asarray(lab).astype(str)
    df = pd.DataFrame({"c": lab, "e": np.asarray(truth)})
    cells = df.value_counts().rename("n").reset_index()
    size_c = cells.groupby("c")["n"].sum()
    size_e = cells.groupby("e")["n"].sum()
    rep = cells.sort_values(["n", "c"], ascending=[False, True]).drop_duplicates("e").set_index("e")
    core = rep["n"]
    s = size_e.reindex(core.index) - core
    m = size_c.reindex(rep["c"]).to_numpy() - core.to_numpy()
    count_err = float(np.abs(m - s.to_numpy()).sum() / size_e.sum())
    tp, pp, tt = _c2(cells["n"]), _c2(size_c), _c2(size_e)
    p, r = (tp / pp if pp else 1.0), (tp / tt if tt else 1.0)
    bmd = int((cells.groupby("c").size() - 1).sum() + (cells.groupby("e").size() - 1).sum())
    rng = np.random.default_rng(0)
    side = rng.random(len(lab)) < 0.5
    cR = df[side].value_counts().rename("r")
    cS = df[~side].value_counts().rename("s")
    both = pd.concat([cR, cS], axis=1).fillna(0)
    TP = float((both["r"] * both["s"]).sum())
    same_c = float((df[side].groupby("c").size() * df[~side].groupby("c").size()).fillna(0).sum())
    same_e = float((df[side].groupby("e").size() * df[~side].groupby("e").size()).fillna(0).sum())
    FP, FN = same_c - TP, same_e - TP
    return dict(dataset=label, method=name, pairF1=2 * p * r / (p + r) if p + r else 0.0, BMD=bmd,
                COUNT_err=count_err, CD_err=abs(len(size_c) - len(size_e)) / len(size_e),
                JOIN_FP_rate=FP / max(TP + FP, 1), JOIN_FN_rate=FN / max(TP + FN, 1), seconds=sec)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--eps", type=float, default=0.2)
    args = ap.parse_args()
    allrows = []
    for label, path in (("MusicBrainz-20K", os.path.join(DATA, "musicbrainz20k", "musicbrainz-20-A01.csv.dapo")),
                        ("MusicBrainz-200K", os.path.join(DATA, "musicbrainz200k", "musicbrainz-200-A01.csv.dapo"))):
        rows, tim, meta = run(path, args.k, args.eps, label)
        allrows += rows
        print(f"\n[{label}] {meta}")
        print("stage seconds: " + ", ".join(f"{k}={v:.1f}" for k, v in tim.items()), flush=True)
    res = pd.DataFrame(allrows)
    res.to_csv(os.path.join(os.path.dirname(HERE), "gpu_out", "t1_ef.csv"), index=False)
    print("\n=== E-F: operator errors and total seconds at scale ===")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
