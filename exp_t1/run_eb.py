"""E-B: closure asymmetry (paper §IV) by controlled edge errors.

Value-level similarity graph (hybrid, eps): nodes = distinct (key value, entity) combinations weighted by
tuple counts; an edge joins two nodes whose key values are eps-similar. Identical values of different
entities are joined (similarity 1), as any value-level method must.
  G_in  : intra-entity edges only (collapse-free baseline, Lemma 7)
  merge : add k inter-entity edges, either the k most similar inter-entity value pairs not yet joined
          ("next", what a looser threshold adds first) or k uniformly random inter-entity pairs ("random")
  split : remove k random intra-entity edges from G_in
Measured per error event: tuple mass moved into foreign classes (impurity) or out of the entity's main class,
and the share of removed edges that were bridges of their entity subgraph.

Usage: python -P exp_t1/run_eb.py [--trials 20]
"""
import argparse
import os
import sys

import networkx as nx
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import datasets as D             # noqa: E402
import grouping as G             # noqa: E402

KS = [1, 2, 5, 10, 20]


def build(df, eps):
    combos = df.groupby(["K", "E"]).size().reset_index(name="w")
    vals = list(dict.fromkeys(combos["K"].astype(str)))
    vi = {v: i for i, v in enumerate(vals)}
    S = G._sim(vals, "hybrid", G.MODEL)
    idx = combos["K"].astype(str).map(vi).to_numpy()
    ent = combos["E"].astype(str).to_numpy()
    w = combos["w"].to_numpy()
    Sn = S[np.ix_(idx, idx)]
    iu = np.triu_indices(len(combos), 1)
    sims = Sn[iu]
    same = ent[iu[0]] == ent[iu[1]]
    on = sims >= 1 - eps - 1e-12
    intra = [(int(a), int(b)) for a, b, s, o in zip(iu[0], iu[1], same, on) if s and o]
    inter_on = [(int(a), int(b)) for a, b, s, o in zip(iu[0], iu[1], same, on) if (not s) and o]
    off = ~on & ~same
    a_off, b_off = iu[0][off], iu[1][off]
    order = np.argsort(-sims[off])[:200]
    cand_next = [(int(a_off[i]), int(b_off[i])) for i in order]
    return ent, w, intra, inter_on, cand_next, iu, same


def impurity_and_split(g, ent, w):
    comp = np.empty(len(ent), dtype=int)
    for ci, cc in enumerate(nx.connected_components(g)):
        for v in cc:
            comp[v] = ci
    t = pd.DataFrame({"c": comp, "e": ent, "w": w}).pivot_table(index="c", columns="e", values="w",
                                                                  aggfunc="sum", fill_value=0)
    v = t.to_numpy()
    merge_mass = int((v.sum(1) - v.max(1)).sum())
    split_mass = int((v.sum(0) - v.max(0)).sum())
    merged_entities = int(((v > 0).sum(1)[(v > 0).sum(1) > 1]).sum())
    return merge_mass, split_mass, merged_entities


def run(df, name, eps, trials, rng):
    ent, w, intra, inter_on, cand_next, iu, same = build(df, eps)
    n = int(w.sum())
    g_in = nx.Graph()
    g_in.add_nodes_from(range(len(ent)))
    g_in.add_edges_from(intra)
    m0, s0, _ = impurity_and_split(g_in, ent, w)
    bridges = set(map(frozenset, nx.bridges(g_in)))
    bridge_share = len(bridges) / max(1, len(intra))
    rows = []
    inter_all = np.flatnonzero(~same)
    for k in KS:
        for mode in ("next", "random"):
            mm = []
            for _ in range(trials):
                g = g_in.copy()
                if mode == "next":
                    pool = cand_next[:max(k, 1) * 3]
                    pick = [pool[i] for i in rng.choice(len(pool), size=min(k, len(pool)), replace=False)]
                else:
                    sel = rng.choice(inter_all, size=k, replace=False)
                    pick = [(int(iu[0][i]), int(iu[1][i])) for i in sel]
                g.add_edges_from(pick)
                mm.append(impurity_and_split(g, ent, w)[0] - m0)
            rows.append(dict(dataset=name, k=k, error=f"merge-{mode}", mass=np.mean(mm) / n,
                             mass_per_event=np.mean(mm) / k / n))
        ss, br = [], []
        for _ in range(trials):
            g = g_in.copy()
            sel = rng.choice(len(intra), size=min(k, len(intra)), replace=False)
            removed = [intra[i] for i in sel]
            br.append(np.mean([frozenset(e) in bridges for e in removed]))
            g.remove_edges_from(removed)
            ss.append(impurity_and_split(g, ent, w)[1] - s0)
        rows.append(dict(dataset=name, k=k, error="split", mass=np.mean(ss) / n,
                         mass_per_event=np.mean(ss) / k / n, bridge_share_removed=np.mean(br)))
    meta = dict(dataset=name, n=n, nodes=len(ent), intra_edges=len(intra), inter_edges_at_eps=len(inter_on),
                bridge_share=bridge_share, base_split=s0 / n, base_merge=m0 / n)
    return rows, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--eps", type=float, default=0.2)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    rows, metas = [], []
    for name, loader in [("Hard-2000", lambda: D.hard(0, 0.2)), ("Hospital", D.hospital), ("Beers", D.beers),
                         ("MusicBrainz", lambda: D.musicbrainz(0, 1000)), ("Settlements", lambda: D.settlements(0, 400)),
                         ("Affiliations", D.affiliations)]:
        df, _ = loader()
        G.clear_cache()
        r, m = run(df, name, args.eps, args.trials, rng)
        rows += r
        metas.append(m)
        print(f"[{name}] done", flush=True)
    res = pd.DataFrame(rows)
    out = os.path.join(os.path.dirname(HERE), "gpu_out", "t1_eb.csv")
    res.to_csv(out, index=False)
    print("\n=== graph structure at eps (value-level hybrid) ===")
    print(pd.DataFrame(metas).to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("\n=== tuple mass moved per injected error event (fraction of n), mean over trials ===")
    piv = res.pivot_table(index=["dataset", "k"], columns="error", values="mass_per_event")
    print(piv.to_string(float_format=lambda v: f"{v:.4f}"))
    print("\n=== share of removed intra edges that were bridges ===")
    print(res[res.error == "split"].pivot_table(index="dataset", columns="k",
                                                values="bridge_share_removed").to_string(float_format=lambda v: f"{v:.2f}"))
    print("\n=== ratio: merge mass per event / split mass per event (k = 1 and 5) ===")
    for k in (1, 5):
        sub = piv.xs(k, level="k")
        print(f"k={k}: " + ", ".join(f"{d}: next {sub.loc[d, 'merge-next'] / max(sub.loc[d, 'split'], 1e-9):.1f}x, "
                                       f"random {sub.loc[d, 'merge-random'] / max(sub.loc[d, 'split'], 1e-9):.1f}x"
                                       for d in sub.index))


if __name__ == "__main__":
    main()
