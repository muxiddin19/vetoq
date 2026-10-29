"""Uniform loaders for T1 experiments.

Each loader returns a DataFrame with columns
  K    dirty grouping key (string)
  E    ground-truth entity id (string)
  CTX  context value (string), or (lat, lon) via columns LAT, LON for geo context; may be absent
and a dict of metadata (ctx_kind in {None, "str", "geo"}).

Downloaded benchmark files under data_t1/ are read as data only (CSV/JSON); nothing in them is executed.
"""
import json
import os
import random

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data_t1")


def hard(seed, rate):
    import sys
    sys.path.insert(0, ROOT)
    from hard_bench import make_clean, corrupt, RATES
    rng = random.Random(seed)
    clean = make_clean(rng)
    out = None
    for r in RATES:                                   # same RNG call order as E13/E18
        raw = corrupt(rng, clean, r)
        if r == rate:
            out = raw
    df = pd.DataFrame({"K": out["country"].astype(str), "E": clean["country"].astype(str),
                       "CTX": out["capital"].astype(str)})
    return df, {"ctx_kind": "str", "name": f"Hard-2000 s{seed} r{rate}"}


def hospital():
    import sys
    sys.path.insert(0, ROOT)
    import exp2_data as E
    d, c = E.load("hospital")
    df = pd.DataFrame({"K": d["HospitalName"], "E": c["ProviderNumber"], "CTX": d["State"]})
    return df, {"ctx_kind": "str", "name": "Hospital"}


def beers():
    import sys
    sys.path.insert(0, ROOT)
    import exp2_data as E
    d, c = E.load("beers")
    df = pd.DataFrame({"K": d["brewery-name"], "E": c["brewery_id"], "CTX": d["city"]})
    return df, {"ctx_kind": "str", "name": "Beers"}


def musicbrainz(seed, n_clusters=1000):
    path = os.path.join(DATA, "musicbrainz20k", "musicbrainz-20-A01.csv.dapo")
    d = pd.read_csv(path, dtype=str, keep_default_na=False)
    rng = np.random.default_rng(seed)
    cids = d["CID"].unique()
    pick = set(rng.choice(cids, size=min(n_clusters, len(cids)), replace=False))
    d = d[d["CID"].isin(pick)].reset_index(drop=True)
    df = pd.DataFrame({"K": d["title"], "E": d["CID"], "CTX": d["artist"]})
    return df, {"ctx_kind": "str", "name": f"MusicBrainz s{seed}"}


def _jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def settlements(seed, n_clusters=400):
    base = os.path.join(DATA, "geosettlements", "x", "geographicalSettelments")
    recs = _jsonl(os.path.join(base, "settlements.json"))
    clusters = _jsonl(os.path.join(base, "combinedSettlements(PerfectMatch).json"))
    ent = {}
    for cl in clusters:
        for v in cl["data"]["clusteredVertices"]:
            ent[v] = f"c{cl['id']}"
    rows = []
    for r in recs:
        dd = r["data"]
        rows.append({"id": r["id"], "K": dd.get("label", ""), "E": ent.get(r["id"], f"s{r['id']}"),
                     "LAT": dd.get("lat", np.nan), "LON": dd.get("lon", np.nan)})
    d = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    multi = d["E"].value_counts()
    multi = multi[multi > 1].index.to_numpy()          # entities with duplicates
    pick = set(rng.choice(multi, size=min(n_clusters, len(multi)), replace=False))
    d = d[d["E"].isin(pick)].reset_index(drop=True)
    return d[["K", "E", "LAT", "LON"]], {"ctx_kind": "geo", "name": f"Settlements s{seed}"}


def affiliations():
    base = os.path.join(DATA, "affiliations", "x")
    ids = pd.read_csv(os.path.join(base, "affiliationstrings_ids.csv"), dtype=str, keep_default_na=False)
    mp = pd.read_csv(os.path.join(base, "affiliationstrings_mapping.csv"), dtype=str, header=None,
                     names=["a", "b"], keep_default_na=False)
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(mp["a"], mp["b"]):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    df = pd.DataFrame({"K": ids["affil1"], "E": [find(i) for i in ids["id1"]]})
    return df, {"ctx_kind": None, "name": "Affiliations"}
