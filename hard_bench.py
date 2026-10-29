"""E13 -- a harder benchmark that separates the methods (2026-10-16).

Why: on Toy-800 / TPC-H-1500 every noise variant is within a small edit
distance of its clean value, so "normalize + approximate FD / DD" and an LLM
match ε-SFD.  This benchmark adds the two cases where similarity semantics
matter:
  * synonyms / abbreviations (USA ~ United States, NYC ~ New York, Wien ~
    Vienna) that no edit-distance threshold merges, and
  * look-alike distinct entities (IRAN/IRAQ, Austria/Australia, Niger/Nigeria,
    Slovakia/Slovenia) that loose thresholds wrongly merge.

Schema: name, city, country, capital, continent, subregion, phone_cc.
Candidates (truth): city->country, country->continent, phone_cc->country (true);
continent->country, name->city, subregion->country (false).  subregion->country
is a trap: every multi-country subregion contains exactly one look-alike pair,
so a method that merges look-alikes accepts it.

Settings fixed before running (all reported): ε = 0.20, τ = 0.10; AFD error
0.05 / 0.10; Split edit limits 2 / 3 / 4; noise rates 0 / 10 / 20 / 30 %;
seeds 0-4; 2000 rows.

Usage:  python hard_bench.py [--llm] [--seeds 0 1 2 3 4] [--model MINILM]
"""

import argparse
import random
import time

import numpy as np
import pandas as pd

# country: (capital, continent, subregion, phone_cc, cities)
COUNTRIES = {
    "United States": ("Washington", "North America", "Northern America", "1",
                      ["New York", "Los Angeles", "San Francisco", "Philadelphia", "Boston", "Houston"]),
    "United Kingdom": ("London", "Europe", "British Isles", "44", ["London", "Manchester"]),
    "Iran": ("Tehran", "Asia", "Middle East", "98", ["Tehran", "Isfahan"]),
    "Iraq": ("Baghdad", "Asia", "Middle East", "964", ["Baghdad", "Basra"]),
    "Austria": ("Vienna", "Europe", "Alpine Europe", "43", ["Vienna", "Salzburg"]),
    "Australia": ("Canberra", "Oceania", "Australasia", "61", ["Sydney", "Melbourne"]),
    "Niger": ("Niamey", "Africa", "West Africa", "227", ["Niamey", "Zinder"]),
    "Nigeria": ("Abuja", "Africa", "West Africa", "234", ["Lagos", "Kano"]),
    "Slovakia": ("Bratislava", "Europe", "Central Europe", "421", ["Bratislava", "Kosice"]),
    "Slovenia": ("Ljubljana", "Europe", "Central Europe", "386", ["Ljubljana", "Maribor"]),
    "Sweden": ("Stockholm", "Europe", "Scandinavia", "46", ["Stockholm", "Gothenburg"]),
    "Switzerland": ("Bern", "Europe", "Western Europe", "41", ["Zurich", "Geneva"]),
}
SYNONYMS = {
    "United States": ["USA", "U.S.", "United States of America", "US"],
    "United Kingdom": ["UK", "U.K.", "Great Britain", "Britain"],
    "New York": ["NYC", "New York City"],
    "Los Angeles": ["LA", "L.A."],
    "San Francisco": ["SF", "San Fran"],
    "Philadelphia": ["Philly"],
    "Washington": ["Washington, D.C.", "Washington DC"],
    "Vienna": ["Wien"],
    "Gothenburg": ["Goteborg"],
    "Kosice": ["Košice"],
    "Zurich": ["Zürich"],
    "North America": ["N. America"],
}
FIRST = ["James", "Mary", "Wookey", "Linda", "Robert", "Susan", "David", "Karen", "Ali", "Ngozi"]
LAST = ["Kim", "Lee", "Park", "Smith", "Garcia", "Chen", "Mueller", "Okafor", "Rahimi", "Novak"]
TEXT_COLS = ["name", "city", "country", "capital", "continent", "subregion"]
CANDS = [("city", "country", True), ("country", "continent", True), ("phone_cc", "country", True),
         ("continent", "country", False), ("name", "city", False), ("subregion", "country", False)]
RATES = (0.0, 0.1, 0.2, 0.3)
EPS, TAU = 0.20, 0.10


def make_clean(rng, n=2000):
    rows = []
    for _ in range(n):
        country = rng.choice(list(COUNTRIES))
        capital, continent, sub, cc, cities = COUNTRIES[country]
        rows.append({"name": f"{rng.choice(FIRST)} {rng.choice(LAST)}", "city": rng.choice(cities),
                     "country": country, "capital": capital, "continent": continent,
                     "subregion": sub, "phone_cc": cc})
    return pd.DataFrame(rows)


def _typo(rng, s):
    """Same seven noise operators as semantic_fd_toy.NOISERS."""
    op = rng.randrange(7)
    if op == 0 and len(s) > 2:                      # adjacent swap
        i = rng.randrange(len(s) - 1)
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op == 1 and len(s) > 2:                      # drop a character
        i = rng.randrange(len(s))
        return s[:i] + s[i + 1:]
    if op == 2:
        return s.upper()
    if op == 3:
        return s.lower()
    if op == 4:
        return s.replace(" ", "")
    if op == 5:
        return s + "."
    if op == 6:
        return " " + s
    return s


def corrupt(rng, df, rate):
    """Each text cell is corrupted with probability `rate`; a corrupted cell
    gets a synonym if one exists (half the time), otherwise a typo."""
    out = df.copy()
    for col in ("city", "country", "capital", "continent", "subregion"):
        vals = out[col].tolist()
        for i, v in enumerate(vals):
            if rng.random() < rate:
                if v in SYNONYMS and rng.random() < 0.5:
                    vals[i] = rng.choice(SYNONYMS[v])
                else:
                    vals[i] = _typo(rng, v)
        out[col] = vals
    return out


def violation_rate(df, x, y):
    g = df.groupby(x)[y].nunique()
    return float((g > 1).mean())


def norm(v):
    return " ".join(str(v).strip().lower().strip(".").split())


# ---------------------------------------------------------------- ε-SFD
def sim_matrix(values, backend, model):
    from difflib import SequenceMatcher
    nv = [norm(v) for v in values]
    n = len(nv)
    S = np.eye(n)
    if backend in ("edit", "hybrid"):
        for i in range(n):
            for j in range(i + 1, n):
                S[i, j] = S[j, i] = SequenceMatcher(None, nv[i], nv[j]).ratio()
    if backend in ("emb", "hybrid"):
        from embedding_backend import embed
        V = embed(nv, model_name=model)
        E = np.clip(V @ V.T, 0, 1)
        np.fill_diagonal(E, 1.0)
        S = np.maximum(S, E) if backend == "hybrid" else E
    return S


def components(S, eps):
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    return connected_components(csr_matrix(S >= 1 - eps), directed=False)[1]


def canon(df, col, backend, model, ctx_col=None):
    """Per-row class labels for `col`; with ctx_col, Definition 8's veto
    sim = min(sim(value), sim(context)) over distinct (value, context) combos."""
    if ctx_col is None:
        uniq = df[col].astype(str).unique().tolist()
        lab = dict(zip(uniq, components(sim_matrix(uniq, backend, model), EPS)))
        return df[col].astype(str).map(lab)
    combos = list(dict.fromkeys(zip(df[col].astype(str), df[ctx_col].astype(str))))
    vals = list(dict.fromkeys(c[0] for c in combos))
    ctxs = list(dict.fromkeys(c[1] for c in combos))
    Sv, Sc = sim_matrix(vals, backend, model), sim_matrix(ctxs, backend, model)
    vi = {v: i for i, v in enumerate(vals)}
    ci = {c: i for i, c in enumerate(ctxs)}
    a = np.array([vi[c[0]] for c in combos])
    b = np.array([ci[c[1]] for c in combos])
    S = np.minimum(Sv[np.ix_(a, a)], Sc[np.ix_(b, b)])
    lab = dict(zip(combos, components(S, EPS)))
    return pd.Series([lab[k] for k in zip(df[col].astype(str), df[ctx_col].astype(str))], index=df.index)


def esfd_accepts(df, backend, model, context):
    out = {}
    for x, y, _ in CANDS:
        t = df[[x, y]].copy()
        for c in (x, y):
            if c == "phone_cc":
                continue                                   # identifier: exact
            ctx = "capital" if (context and c == "country" and "capital" not in (x, y)) else None
            t[c] = canon(df, c, backend, model, ctx).values
        out[(x, y)] = violation_rate(t, x, y) <= TAU
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--llm", action="store_true")
    ap.add_argument("--llm-model", default="claude-opus-5-5")
    ap.add_argument("--only-fastdd", action="store_true",
                    help="evaluate only FastDD (Kuang et al. 2024); skips everything else incl. the LLM")
    ap.add_argument("--only-split", action="store_true",
                    help="evaluate only Split DD; skips everything else incl. the LLM")
    args = ap.parse_args()

    import desbordante as d
    from baselines import accepted_from_fds, run_fd, run_split, run_fastdd, llm_verdicts
    cols = ["name", "city", "country", "capital", "continent", "subregion", "phone_cc"]
    exact_ok = {"phone_cc"}
    anon = {c: chr(ord("A") + i) for i, c in enumerate(cols)}
    cands_anon = [(anon[x], anon[y], t) for x, y, t in CANDS]

    methods = {}
    def add(name, fn):
        methods.setdefault(name, fn)
    for b in ("edit", "hybrid"):
        add(f"ε-SFD {b}", (lambda b: lambda raw, nrm: esfd_accepts(raw, b, args.model, False))(b))
    add("ε-SFD hybrid + Def 8 (country|capital)", lambda raw, nrm: esfd_accepts(raw, "hybrid", args.model, True))
    for tag in ("raw", "norm"):
        pick = (lambda tag: (lambda raw, nrm: raw if tag == "raw" else nrm))(tag)
        add(f"exact FD (HyFD) [{tag}]", (lambda p: lambda raw, nrm: accepted_from_fds(
            run_fd(d.fd.algorithms.HyFD(), p(raw, nrm), {}), cols, CANDS))(pick))
        for e in (0.05, 0.10):
            add(f"TANE AFD g3<={e:.2f} [{tag}]", (lambda p, e: lambda raw, nrm: accepted_from_fds(
                run_fd(d.afd.algorithms.Tane(), p(raw, nrm), {"error": e, "afd_error_measure": "g3"}), cols, CANDS))(pick, e))
            add(f"Pyro AFD <={e:.2f} [{tag}]", (lambda p, e: lambda raw, nrm: accepted_from_fds(
                run_fd(d.afd.algorithms.Pyro(), p(raw, nrm), {"error": e}), cols, CANDS))(pick, e))
        for k in (2, 3, 4):
            add(f"Split DD edit<={k} [{tag}]", (lambda p, k: lambda raw, nrm: run_split(
                p(raw, nrm), cols, CANDS, exact_ok, k)[0])(pick, k))
        for k in (2, 3, 4):
            add(f"FastDD edit<={k} [{tag}]", (lambda p, k: lambda raw, nrm: run_fastdd(
                p(raw, nrm), cols, CANDS, exact_ok, k)[0])(pick, k))
    if args.only_fastdd:
        methods = {m: fn for m, fn in methods.items() if m.startswith("FastDD")}
        args.llm = False
    if args.only_split:
        methods = {m: fn for m, fn in methods.items() if m.startswith("Split")}
        args.llm = False

    tally = {m: [0, 0] for m in methods}
    per_cand = {m: {(x, y): [0, 0] for x, y, _ in CANDS} for m in methods}
    times = {m: 0.0 for m in methods}
    llm_names = []
    client = None
    if args.llm:
        import anthropic
        client = anthropic.Anthropic()
        llm_names = [f"LLM {args.llm_model} (named columns)", f"LLM {args.llm_model} (anonymized A-G)"]
        for m in llm_names:
            tally[m] = [0, 0]
            per_cand[m] = {(x, y): [0, 0] for x, y, _ in CANDS}
            times[m] = 0.0
    tokens = [0, 0]

    for seed in args.seeds:
        rng = random.Random(seed)
        clean = make_clean(rng)
        for rate in RATES:
            raw = corrupt(rng, clean, rate)[cols].astype(str)
            nrm = raw.apply(lambda c: c if c.name in exact_ok else c.map(norm))
            for m, fn in methods.items():
                t0 = time.perf_counter()
                v = fn(raw, nrm)
                times[m] += time.perf_counter() - t0
                for x, y, truth in CANDS:
                    ok = v[(x, y)] == truth
                    tally[m][0] += ok; tally[m][1] += 1
                    per_cand[m][(x, y)][0] += ok; per_cand[m][(x, y)][1] += 1
            for m in llm_names:
                anonym = "anonymized" in m
                table = raw.rename(columns=anon) if anonym else raw
                cands = cands_anon if anonym else CANDS
                t0 = time.perf_counter()
                v, stop, usage = llm_verdicts(client, args.llm_model, "high", table, cands,
                                              "relational table" if anonym else "customer location table")
                times[m] += time.perf_counter() - t0
                tokens[0] += usage.input_tokens; tokens[1] += usage.output_tokens
                if v is None:
                    print(f"  {m} seed {seed} @{rate:.0%}: stop_reason={stop} -> counted wrong")
                for (x, y, truth), (xa, ya, _) in zip(CANDS, cands):
                    ok = (v is not None) and v[(xa, ya)] == truth
                    tally[m][0] += ok; tally[m][1] += 1
                    per_cand[m][(x, y)][0] += ok; per_cand[m][(x, y)][1] += 1
            print(f"[seed {seed} rate {rate:.0%}] done", flush=True)

    short = {("city", "country"): "city>ctry", ("country", "continent"): "ctry>cont",
             ("phone_cc", "country"): "cc>ctry", ("continent", "country"): "cont>ctry*",
             ("name", "city"): "name>city*", ("subregion", "country"): "subr>ctry*"}
    print(f"\nE13 hard benchmark: seeds={args.seeds}, rates={RATES}, eps={EPS}, tau={TAU}, "
          f"emb={args.model}   (* = false FD)")
    hdr = "".join(f"{short[(x, y)]:>11}" for x, y, _ in CANDS)
    print(f"{'method':<44}{'acc':>6}{hdr}{'time(s)':>9}")
    for m in tally:
        h, n = tally[m]
        cells = "".join(f"{per_cand[m][(x, y)][0]:>6}/{per_cand[m][(x, y)][1]:<4}" for x, y, _ in CANDS)
        print(f"{m:<44}{h / n:>6.0%}{cells}{times[m]:>9.1f}")
    if args.llm:
        print(f"\nLLM tokens in/out: {tokens[0]:,} / {tokens[1]:,}")


if __name__ == "__main__":
    main()
