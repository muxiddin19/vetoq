"""E-H analysis: Splink default configuration, best Splink configuration in hindsight, best of the 24 VetoQ
candidates, and the LLM of E-G, all on the same 20 instances (per-instance best, then mean per dataset).

Usage: python -P exp_t1/analyze_eh.py
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
G = os.path.join(ROOT, "gpu_out")
eh = pd.read_csv(os.path.join(G, "t1_eh.csv"))
eg = pd.read_csv(os.path.join(G, "t1_eg_hf_Qwen2.5-1.5B-Instruct.csv"))
ea = pd.read_csv(os.path.join(G, "t1_ea.csv"))
ea = ea[ea.inst.isin(eh.inst)]
n_rows = ea.groupby("inst")["n_rows"].first()
ops = ["COUNT_err", "SUM_err", "CD_err", "JOIN_FP_rate", "JOIN_FN_rate"]


def default_splink(g):
    m = "Splink-em-ctx-0.5" if (g.method == "Splink-em-ctx-0.5").any() else "Splink-em-key-0.5"
    return g[g.method == m].iloc[0]


def llm(g):
    m = "LLM-context" if (g.method == "LLM-context").any() else "LLM-value"
    return g[g.method == m].iloc[0]


rows = []
for inst in sorted(eh.inst.unique()):
    s, a, l = eh[eh.inst == inst], ea[ea.inst == inst], eg[eg.inst == inst]
    d = default_splink(s)
    f1pick = s.loc[s.pairF1.idxmax()]
    line = {"dataset": d["dataset"], "inst": inst, "def:M/n": d["M"] / n_rows[inst], "def:S/n": d["S"] / n_rows[inst],
            "def:pF1": d["pairF1"], "best24:pF1": a.pairF1.max()}
    for o in ops:
        line[f"def:{o}"] = d[o]
        line[f"f1pick:{o}"] = f1pick[o]
        line[f"bestSplink:{o}"] = s[o].min()
        line[f"best24:{o}"] = a[o].min()
        line[f"hardveto:{o}"] = a.loc[a.method == "hybrid-any-0.2+hardveto", o].mean()
        line[f"llm:{o}"] = llm(l)[o]
        line[f"rank_bestSplink:{o}"] = int((a[o] < s[o].min() - 1e-12).sum()) + 1
    rows.append(line)
t = pd.DataFrame(rows)
fmt = lambda v: f"{v:.3f}"  # noqa: E731

print("=== Splink default (EM, context if any, p>=0.5): masses and pairwise F1 ===")
print(t.groupby("dataset")[["def:M/n", "def:S/n", "def:pF1", "best24:pF1"]].mean().to_string(float_format=fmt))
for o in ops:
    cols = [f"def:{o}", f"f1pick:{o}", f"bestSplink:{o}", f"best24:{o}", f"llm:{o}"]
    print(f"\n[{o}]  def = Splink default, f1pick = Splink config with best pairwise F1, bestSplink = best of 12 "
          f"Splink configs for this operator, best24 = best VetoQ candidate, llm = E-G")
    print(t.groupby("dataset")[cols].mean().to_string(float_format=fmt))
print("\nshare of instances where the best Splink configuration beats or ties every VetoQ candidate:")
print((t[[f"rank_bestSplink:{o}" for o in ops]] == 1).mean().to_string(float_format=lambda v: f"{v:.2f}"))
print("\nmean over instances (all datasets):")
print(t[[c for c in t.columns if c.startswith(("def:", "f1pick:", "bestSplink:", "best24:", "llm:"))]].mean()
      .to_string(float_format=fmt))

# fixed equivalences over the 20 instances (paper: summary table)
print("\n=== mean over the 20 instances: fixed equivalences and hindsight bounds (hard veto: instances with context) ===")
summ = {m: ea[ea.method == m][ops].mean() for m in ("exact", "norm", "hybrid-any-0.2+hardveto")}
summ["Splink default"] = t[[f"def:{o}" for o in ops]].mean().set_axis(ops)
summ["LLM"] = t[[f"llm:{o}" for o in ops]].mean().set_axis(ops)
summ["Splink best of 12"] = t[[f"bestSplink:{o}" for o in ops]].mean().set_axis(ops)
summ["best of 24"] = t[[f"best24:{o}" for o in ops]].mean().set_axis(ops)
print(pd.DataFrame(summ).T.to_string(float_format=fmt))

# corrected E-G comparison: best of 24 on exactly the E-G instances
print("\n=== E-G corrected: best of 24 per instance, restricted to E-G instances ===")
print(t.groupby("dataset")[[f"best24:{o}" for o in ops]].mean().to_string(float_format=fmt))
