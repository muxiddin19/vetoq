"""E-H closure check: how many false (inter-entity) record pairs does Splink's connected-components clustering
produce, compared with the false pairs it scored directly above the threshold?

For the default configuration (EM, context, as in run_eh.py) and thresholds 0.1 / 0.3 / 0.5 / 0.9 we report
  scored  = inter-entity pairs with match probability >= threshold (decided by the model)
  closed  = inter-entity pairs inside one cluster after connected components (what the operators see)
  ratio   = closed / scored  (> 1: closure added false pairs that the model never accepted)
  merge   = sum_e m(e) / n

Usage: python -P exp_t1/run_eh_closure.py
"""
import logging
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import contingency as C                                   # noqa: E402
from run_eg import instances                              # noqa: E402
from run_eh import prepare                                # noqa: E402

import splink.comparison_library as cl                    # noqa: E402
from splink import Linker, SettingsCreator, block_on      # noqa: E402
from splink.backends.duckdb import DuckDBAPI              # noqa: E402

THRESHOLDS = (0.1, 0.3, 0.5, 0.9)


def false_pairs_in_clusters(lab, truth):
    n = pd.crosstab(pd.Series(lab, name="c"), pd.Series(truth, name="e")).to_numpy()
    size = n.sum(axis=1)
    return int(((size * (size - 1)) // 2 - ((n * (n - 1)) // 2).sum(axis=1)).sum())


def main():
    logging.getLogger("splink").setLevel(logging.ERROR)
    rows = []
    for loader in instances():
        df, meta = loader()
        if meta["ctx_kind"] is None:
            continue
        d = prepare(df, meta)
        comps = [cl.JaroWinklerAtThresholds("Kn", [0.95, 0.88, 0.8])]
        comps.append(cl.JaroWinklerAtThresholds("Cn", [0.95, 0.88]) if meta["ctx_kind"] == "str"
                     else cl.DistanceInKMAtThresholds("LAT", "LON", [1, 25, 100]))
        settings = SettingsCreator(link_type="dedupe_only", comparisons=comps,
                                   blocking_rules_to_generate_predictions=["1=1"], retain_matching_columns=False)
        linker = Linker(DuckDBAPI().register(d), settings, log_level=None)
        linker.training.estimate_probability_two_random_records_match([block_on("Kn")], recall=0.7)
        linker.training.estimate_u_using_random_sampling(max_pairs=1e6, seed=0)
        linker.training.estimate_parameters_using_expectation_maximisation(block_on("first3"),
                                                                           record_sample_proportion=1.0)
        if meta["ctx_kind"] == "str":
            linker.training.estimate_parameters_using_expectation_maximisation(block_on("Cn"),
                                                                               record_sample_proportion=1.0)
        pred = linker.inference.predict(threshold_match_probability=min(THRESHOLDS))
        p = pred.as_pandas_dataframe()[["unique_id_l", "unique_id_r", "match_probability"]]
        truth = d["E"].to_numpy()
        p["false"] = truth[p.unique_id_l.to_numpy()] != truth[p.unique_id_r.to_numpy()]
        for th in THRESHOLDS:
            c = linker.clustering.cluster_pairwise_predictions_at_threshold(pred, threshold_match_probability=th)
            c = c.as_pandas_dataframe().set_index("unique_id")["cluster_id"].reindex(d["unique_id"])
            lab = c.astype(str).to_numpy()
            scored = int(p.loc[p.match_probability >= th, "false"].sum())
            closed = false_pairs_in_clusters(lab, truth)
            ms = C.masses(C.table(lab, truth))
            rows.append(dict(dataset=meta["name"].split(" ")[0], inst=meta["name"], th=th, scored=scored,
                             closed=closed, merge=float(ms["m"].sum()) / len(d)))
        print(f"  {meta['name']}", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(os.path.join(ROOT, "gpu_out", "t1_eh_closure.csv"), index=False)
    g = r.groupby(["dataset", "th"])[["scored", "closed", "merge"]].mean()
    g["ratio"] = g["closed"] / g["scored"].replace(0, np.nan)
    print("\n=== Splink default: false pairs scored vs. false pairs after connected components (mean per instance) ===")
    print(g.to_string(float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
