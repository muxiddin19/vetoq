"""E-H: an off-the-shelf ER tool (Splink 5, unsupervised Fellegi-Sunter on DuckDB) versus VetoQ's candidates.

Standard Splink deduplication: Jaro-Winkler levels on the lower-cased key and on the context (distance in km for
Settlements), u from random sampling, lambda from exact key matches (recall 0.7), m by EM, all pairs scored,
connected components at match probability 0.5 / 0.9 / 0.99. Two variants: key only, key + context.
Scored with the metrics and random draws of run_ea.py (via run_eg.score) on the 20 E-G instances.

Usage: python -P exp_t1/run_eh.py
"""
import logging
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from run_eg import instances, score   # noqa: E402

import splink.comparison_library as cl                   # noqa: E402
from splink import Linker, SettingsCreator, block_on     # noqa: E402
from splink.backends.duckdb import DuckDBAPI             # noqa: E402

THRESHOLDS = (0.5, 0.9, 0.99)
SEED = 0


def norm(s):
    s = re.sub(r"[^\w ]", " ", str(s).lower())
    s = " ".join(s.split())
    return s or None


def prepare(df, meta):
    kn = [norm(k) for k in df["K"]]
    d = pd.DataFrame({"unique_id": np.arange(len(df)), "Kn": kn})
    d["first3"] = [k[:3] if k else None for k in kn]
    d["E"] = df["E"].astype(str).to_numpy()   # used only by the label-trained variant
    if meta["ctx_kind"] == "str":
        d["Cn"] = [None if str(c).strip().lower() in ("", "nan", "[unknown]") else norm(c) for c in df["CTX"]]
    elif meta["ctx_kind"] == "geo":
        d["LAT"] = df["LAT"].to_numpy(dtype=float)
        d["LON"] = df["LON"].to_numpy(dtype=float)
    return d


def run_splink(d, ctx_kind, use_ctx, train):
    comps = [cl.JaroWinklerAtThresholds("Kn", [0.95, 0.88, 0.8])]
    if use_ctx and ctx_kind == "str":
        comps.append(cl.JaroWinklerAtThresholds("Cn", [0.95, 0.88]))
    elif use_ctx and ctx_kind == "geo":
        comps.append(cl.DistanceInKMAtThresholds("LAT", "LON", [1, 25, 100]))
    settings = SettingsCreator(link_type="dedupe_only", comparisons=comps,
                               blocking_rules_to_generate_predictions=["1=1"], retain_matching_columns=False)
    db = DuckDBAPI()
    linker = Linker(db.register(d), settings, log_level=None)
    linker.training.estimate_probability_two_random_records_match([block_on("Kn")], recall=0.7)
    linker.training.estimate_u_using_random_sampling(max_pairs=1e6, seed=SEED)
    if train == "em":
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("first3"),
                                                                           record_sample_proportion=1.0)
        if use_ctx and ctx_kind == "str":
            linker.training.estimate_parameters_using_expectation_maximisation(block_on("Cn"),
                                                                               record_sample_proportion=1.0)
    else:   # supervised m from the ground-truth entity column: an upper bound for the tool's model
        linker.training.estimate_m_from_label_column("E")
    pred = linker.inference.predict(threshold_match_probability=min(THRESHOLDS))
    n_pairs = int(pred.as_pandas_dataframe().shape[0])
    out = {}
    for th in THRESHOLDS:
        cl_df = linker.clustering.cluster_pairwise_predictions_at_threshold(pred, threshold_match_probability=th)
        c = cl_df.as_pandas_dataframe().set_index("unique_id")["cluster_id"]
        out[th] = c.reindex(d["unique_id"]).astype(str).to_numpy()
    return out, n_pairs


def main():
    logging.getLogger("splink").setLevel(logging.ERROR)
    rows = []
    for loader in instances():
        df, meta = loader()
        ds = meta["name"].split(" ")[0]
        d = prepare(df, meta)
        for train in ("em", "labels"):
            for use_ctx in ([False, True] if meta["ctx_kind"] else [False]):
                t0 = time.perf_counter()
                labs, n_pairs = run_splink(d, meta["ctx_kind"], use_ctx, train)
                sec = time.perf_counter() - t0
                for th, lab in labs.items():
                    m = f"Splink-{train}-{'ctx' if use_ctx else 'key'}-{th}"
                    rows.append(dict(dataset=ds, inst=meta["name"], method=m, sec=sec, pairs_kept=n_pairs,
                                     **score(df, meta["name"], lab)))
                print(f"  {meta['name']:28s} {train:6s} ctx={use_ctx!s:5s} {sec:6.1f}s  pairs>=0.5: {n_pairs}",
                      flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ROOT, "gpu_out", "t1_eh.csv"), index=False)

    ea = pd.read_csv(os.path.join(ROOT, "gpu_out", "t1_ea.csv"))
    ea = ea[ea.inst.isin(res.inst)]
    ops = ["COUNT_err", "SUM_err", "CD_err", "JOIN_FP_rate", "JOIN_FN_rate"]
    print("\n=== E-H: mean per dataset (Splink variants; hard veto; best of 24 per instance and operator) ===")
    for o in ops + ["pairF1"]:
        t = res.pivot_table(index="dataset", columns="method", values=o, aggfunc="mean")
        t["hardveto"] = ea[ea.method == "hybrid-any-0.2+hardveto"].groupby("dataset")[o].mean()
        agg = "max" if o == "pairF1" else "min"
        t["best24"] = ea.groupby(["dataset", "inst"])[o].agg(agg).groupby("dataset").mean()
        print(f"\n[{o}]")
        print(t.to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nSplink time: {res.drop_duplicates(['inst', 'method'])['sec'].sum() / len(THRESHOLDS):.0f} s total")


if __name__ == "__main__":
    main()
