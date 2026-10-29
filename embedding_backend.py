"""
embedding_backend.py — Week-6 hypothesis-gate harness (GPU office machine).

THE HYPOTHESIS (working doc, Wk-6 gate):
  An embedding similarity backend can SIMULTANEOUSLY
    (P) merge true synonyms:        sim("NYC", "New York")     >= 1 - eps
    (S) separate distinct entities:  sim("IRAN", "IRAQ")       <  1 - eps
  whereas the edit-distance backend provably cannot (Theorem/Prop 3:
  edit-sim(IRAN,IRNA)=edit-sim(IRAN,IRAQ)=0.75 — no threshold separates).

This script is the drop-in semantic backend for semantic_fd_toy.py /
semantic_fd_tpch.py. It replaces the SequenceMatcher similarity inside
canonicalize() with sentence-embedding cosine.

RUN ORDER ON THE GPU BOX (see GPU_EXPERIMENT_GUIDE.md):
  1. python embedding_backend.py --selftest      # P/S gate, no FD logic
  2. patch canonicalize() to import sim_embed     # one-line seam
  3. rerun semantic_fd_tpch.py                    # full FD accuracy
"""

import argparse
import numpy as np

# ---- model loaded lazily so --help works without torch installed ----------
_MODEL = None
_CACHE = {}


def _load(model_name):
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer  # GPU box only
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[embedding_backend] loading {model_name} on {dev}")
        _MODEL = SentenceTransformer(model_name, device=dev)
    return _MODEL


def embed(texts, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    m = _load(model_name)
    todo = [t for t in texts if t not in _CACHE]
    if todo:
        vecs = m.encode(todo, normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=256)
        _CACHE.update(dict(zip(todo, vecs)))
    return np.array([_CACHE[t] for t in texts])


def sim_embed(a: str, b: str) -> float:
    """Cosine of normalized embeddings -> [−1,1]; clamp to [0,1].
    Drop-in replacement for SequenceMatcher(None,a,b).ratio()."""
    if a == b:
        return 1.0
    va, vb = embed([a, b])
    return float(max(0.0, np.dot(va, vb)))


# ----------------------------------------------------------------- self-test
SYNONYM_PAIRS = [          # must MERGE  (high sim)
    ("NYC", "New York"), ("NYC", "New York City"),
    ("USA", "United States"), ("UK", "United Kingdom"),
    ("LA", "Los Angeles"), ("SF", "San Francisco"),
]
DISTINCT_PAIRS = [         # must SEPARATE (low sim) — the edit-distance trap
    ("IRAN", "IRAQ"), ("Austria", "Australia"),
    ("Slovakia", "Slovenia"), ("Niger", "Nigeria"),
    ("Sweden", "Switzerland"), ("Chile", "China"),
]


PROP3_PAIRS = [            # §6.4 / Prop 3 diagnostic (not part of the gate)
    ("IRAN", "IRNA"), ("IRAN", "IRN"), ("IRAN", "IRAQ"),
]


def prop3_probe():
    from difflib import SequenceMatcher
    print("\n--- Prop 3 probe (TPC-H ambiguity zone; not part of the gate) ---")
    print(f"  {'pair':<14} {'edit-ratio':>10} {'emb sim':>8} {'emb dist':>9}")
    for a, b in PROP3_PAIRS:
        e = SequenceMatcher(None, a.lower(), b.lower()).ratio()
        s = sim_embed(a, b)
        print(f"  {a + '~' + b:<14} {e:>10.3f} {s:>8.3f} {1 - s:>9.3f}")


def selftest(model_name, eps_grid=(0.2, 0.3, 0.4, 0.5)):
    _load(model_name)
    syn = [(a, b, sim_embed(a, b)) for a, b in SYNONYM_PAIRS]
    dis = [(a, b, sim_embed(a, b)) for a, b in DISTINCT_PAIRS]
    print("\n--- synonym pairs (want HIGH sim) ---")
    for a, b, s in syn:
        print(f"  {a:>6} ~ {b:<16} sim={s:.3f}")
    print("--- distinct pairs (want LOW sim) ---")
    for a, b, s in dis:
        print(f"  {a:>9} ~ {b:<13} sim={s:.3f}")

    syn_min = min(s for *_, s in syn)
    dis_max = max(s for *_, s in dis)
    print(f"\nmin synonym sim = {syn_min:.3f}   "
          f"max distinct sim = {dis_max:.3f}")
    separable = syn_min > dis_max
    print(f"GLOBAL SEPARABILITY (a single eps splits all): "
          f"{'YES' if separable else 'NO'}  "
          f"(gap = {syn_min - dis_max:+.3f})")

    print("\nPer-eps confusion (merge if sim >= 1-eps):")
    print(f"{'eps':>5} | {'thr':>5} | {'syn merged':>10} | {'dis WRONGLY merged':>18}")
    for eps in eps_grid:
        thr = 1 - eps
        sm = sum(s >= thr for *_, s in syn)
        dm = sum(s >= thr for *_, s in dis)
        print(f"{eps:>5} | {thr:>5.2f} | {sm:>3}/{len(syn):<6} | {dm:>3}/{len(dis):<14}")

    print("\nHYPOTHESIS GATE:")
    if separable:
        print("  PASS — embeddings achieve P and S together. "
              "Core thesis (C1) empirically supported.")
    else:
        print("  PARTIAL/FAIL — no global threshold separates. "
              "Escalate to context-aware equivalence (Def 8, working doc) "
              "and report the per-eps tradeoff curve as the finding.")
    prop3_probe()
    return separable


# ------------------------------------------------- E12a: context-aware gate
# Each pair carries a context value per side (capital / state). Contexts
# include real spelling variants (Teheran, Wien, NY, CA) so that the context
# attribute is itself noisy. Controlled gate, not a benchmark.
CTX_MERGE = [              # must MERGE: synonyms + typos, same real entity
    ("NYC", "NY", "New York", "New York"),
    ("NYC", "NY", "New York City", "New York"),
    ("USA", "Washington", "United States", "Washington, D.C."),
    ("UK", "London", "United Kingdom", "London"),
    ("LA", "CA", "Los Angeles", "California"),
    ("SF", "California", "San Francisco", "CA"),
    ("IRAN", "Tehran", "IRNA", "Teheran"),
    ("IRAN", "Tehran", "IRN", "Tehran"),
    ("Austria", "Vienna", "Austira", "Wien"),
    ("Slovenia", "Ljubljana", "Slovnia", "Ljubljana"),
]
CTX_SEPARATE = [           # must SEPARATE: distinct entities
    ("IRAN", "Tehran", "IRAQ", "Baghdad"),
    ("Austria", "Vienna", "Australia", "Canberra"),
    ("Slovakia", "Bratislava", "Slovenia", "Ljubljana"),
    ("Niger", "Niamey", "Nigeria", "Abuja"),
    ("Sweden", "Stockholm", "Switzerland", "Bern"),
    ("Chile", "Santiago", "China", "Beijing"),
]


def sim_edit(a: str, b: str) -> float:
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _report(label, merge, sep, eps_grid):
    m_min, s_max = min(merge), max(sep)
    print(f"\n[{label}]  min must-merge = {m_min:.3f}   "
          f"max must-separate = {s_max:.3f}   gap = {m_min - s_max:+.3f}   "
          f"{'SEPARABLE' if m_min > s_max else 'not separable'}")
    print(f"  {'eps':>5} | {'merged':>8} | {'WRONGLY merged':>14}")
    for eps in eps_grid:
        thr = 1 - eps
        print(f"  {eps:>5} | {sum(s >= thr for s in merge):>3}/{len(merge):<4} | "
              f"{sum(s >= thr for s in sep):>3}/{len(sep):<10}")
    return m_min - s_max


def context_gate(model_name, eps_grid=(0.1, 0.2, 0.3, 0.4, 0.5)):
    """E12a. Value-only backends vs Def 8 context veto
    sim_ctx = min(sim(v1,v2), sim(c1,c2)) vs serialized-embedding baseline
    cos(emb('v1 | c1'), emb('v2 | c2'))."""
    _load(model_name)
    backends = {
        "edit": sim_edit,
        "emb": sim_embed,
        "max(edit,emb)": lambda a, b: max(sim_edit(a, b), sim_embed(a, b)),
    }

    print("\n--- per-pair similarities ---")
    print(f"  {'pair':<36} {'edit':>6} {'emb':>6} {'max':>6} "
          f"{'ctx(max)':>8} {'veto':>6} {'serial':>7}")
    for kind, pairs in (("M", CTX_MERGE), ("S", CTX_SEPARATE)):
        for v1, c1, v2, c2 in pairs:
            se, sm = sim_edit(v1, v2), sim_embed(v1, v2)
            sx = max(se, sm)
            cx = max(sim_edit(c1, c2), sim_embed(c1, c2))
            sr = sim_embed(f"{v1} | {c1}", f"{v2} | {c2}")
            name = f"{kind} {v1}({c1})~{v2}({c2})"
            print(f"  {name:<36} {se:>6.3f} {sm:>6.3f} {sx:>6.3f} "
                  f"{cx:>8.3f} {min(sx, cx):>6.3f} {sr:>7.3f}")

    gaps = {}
    for bname, sim in backends.items():
        merge = [sim(v1, v2) for v1, _, v2, _ in CTX_MERGE]
        sep = [sim(v1, v2) for v1, _, v2, _ in CTX_SEPARATE]
        gaps[f"{bname} / value only"] = _report(f"{bname} / value only", merge, sep, eps_grid)
        merge = [min(sim(v1, v2), sim(c1, c2)) for v1, c1, v2, c2 in CTX_MERGE]
        sep = [min(sim(v1, v2), sim(c1, c2)) for v1, c1, v2, c2 in CTX_SEPARATE]
        gaps[f"{bname} / Def 8 veto"] = _report(f"{bname} / Def 8 veto", merge, sep, eps_grid)
    merge = [sim_embed(f"{v1} | {c1}", f"{v2} | {c2}") for v1, c1, v2, c2 in CTX_MERGE]
    sep = [sim_embed(f"{v1} | {c1}", f"{v2} | {c2}") for v1, c1, v2, c2 in CTX_SEPARATE]
    gaps["serialized emb (baseline)"] = _report("serialized emb (baseline)", merge, sep, eps_grid)

    print("\n--- summary (gap = min must-merge - max must-separate) ---")
    for k, g in gaps.items():
        print(f"  {k:<28} gap = {g:+.3f}  {'SEPARABLE' if g > 0 else ''}")
    return gaps


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--context", action="store_true",
                    help="E12a: context-aware (Def 8) gate")
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    args = ap.parse_args()
    if args.context:
        context_gate(args.model)
    elif args.selftest:
        selftest(args.model)
    else:
        print("Use --selftest, or import sim_embed into canonicalize().")
