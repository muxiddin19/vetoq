"""E-G analysis: merge/split masses of the LLM equivalences, and where they rank among the 24 VetoQ candidates.

Usage: python -P exp_t1/analyze_eg.py [gpu_out/t1_eg_hf_Qwen2.5-1.5B-Instruct.csv]
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "gpu_out", "t1_eg_hf_Qwen2.5-1.5B-Instruct.csv")
eg = pd.read_csv(path)
ea = pd.read_csv(os.path.join(ROOT, "gpu_out", "t1_ea.csv"))
n_rows = ea.groupby("inst")["n_rows"].first()
eg["n_rows"] = eg["inst"].map(n_rows)
ops = ["COUNT_err", "SUM_err", "CD_err", "JOIN_FP_rate", "JOIN_FN_rate"]

print("=== masses (fraction of n) and pairwise F1, mean per dataset ===")
both = pd.concat([eg, ea[ea.method.isin(["norm", "hybrid-any-0.2+hardveto"])]])
both["M/n"] = both["M"] / both["n_rows"]
both["S/n"] = both["S"] / both["n_rows"]
print(both.groupby(["dataset", "method"])[["pairF1", "M/n", "S/n"]].mean().to_string(float_format=lambda v: f"{v:.3f}"))

print("\n=== rank of LLM-context (else LLM-value) among the 24 candidates + itself; 1 = best (ties count as best) ===")
rows = []
for inst, g in eg.groupby("inst"):
    llm = g[g.method == "LLM-context"] if (g.method == "LLM-context").any() else g[g.method == "LLM-value"]
    llm = llm.iloc[0]
    cand = ea[ea.inst == inst]
    line = {"dataset": llm["dataset"], "inst": inst}
    for o in ops:
        line[o] = int((cand[o] < llm[o] - 1e-12).sum()) + 1
    rows.append(line)
rk = pd.DataFrame(rows)
print(rk.groupby("dataset")[ops].mean().to_string(float_format=lambda v: f"{v:.1f}"))
print("\nshare of instances where the LLM equivalence is best (rank 1):")
print((rk[ops] == 1).mean().to_string(float_format=lambda v: f"{v:.2f}"))
print(f"instances: {len(rk)}")

print("\n=== per instance: is there one equivalence (of 25) best for all operators at once? ===")
cnt = 0
for inst, g in eg.groupby("inst"):
    llm = g[g.method == "LLM-context"] if (g.method == "LLM-context").any() else g[g.method == "LLM-value"]
    pool = pd.concat([ea[ea.inst == inst], llm])
    winners = [set(pool.loc[pool[o] <= pool[o].min() + 1e-12, "method"]) for o in ops]
    cnt += bool(set.intersection(*winners))
print(f"instances with a common best equivalence for all five operators: {cnt} of {eg.inst.nunique()}")
