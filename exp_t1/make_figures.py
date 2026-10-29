"""Figures for the T1 paper (PDF for LaTeX, PNG for preview). Reads gpu_out/t1_*.csv.

Fig. 1  threshold/linkage trade-off: total over-count vs under-count of the hybrid (E-A data)
Fig. 2  disagreement between ER metrics and operator errors (E-A data)
Usage: python -P exp_t1/make_figures.py
"""
import itertools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402
import pandas as pd              # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures_t1")
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "legend.fontsize": 6.5, "pdf.fonttype": 42})

ea = pd.read_csv(os.path.join(ROOT, "gpu_out", "t1_ea.csv"))


def save(fig, name):
    fig.savefig(os.path.join(OUT, f"{name}.pdf"))
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=200)


def fig_tradeoff():
    h = ea[ea.method.str.match(r"^hybrid-(any|all)-0\.[123]$")].copy()
    h["link"] = h.method.str.split("-").str[1]
    h["eps"] = h.method.str.split("-").str[2].astype(float)
    h["over"] = h["M"] / h["n_rows"]
    h["under"] = h["S"] / h["n_rows"]
    t = h.groupby(["dataset", "link", "eps"])[["over", "under"]].mean().reset_index()
    sets = ["Hard-2000", "Hospital", "Settlements", "Affiliations", "MusicBrainz"]
    markers = dict(zip(sets, "osD^v"))
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for ds in sets:
        for link, ls in (("any", "-"), ("all", "--")):
            g = t[(t.dataset == ds) & (t.link == link)].sort_values("eps")
            ax.plot(g["under"] + 1e-3, g["over"] + 1e-3, ls, marker=markers[ds], ms=3, lw=1,
                    color=f"C{sets.index(ds)}", label=ds if link == "any" else None)
            # mark the eps = 0.3 end of each curve
            last = g.iloc[-1]
            ax.plot(last["under"] + 1e-3, last["over"] + 1e-3, marker=markers[ds], ms=5,
                    mfc="none", color=f"C{sets.index(ds)}", lw=0)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"total under-count $\Sigma_e s(e)/n$ (+0.001)")
    ax.set_ylabel(r"total over-count $\Sigma_e m(e)/n$ (+0.001)")
    ax.set_title("solid: any-linkage, dashed: all-linkage")
    ax.legend(loc="lower left", frameon=False)
    fig.tight_layout()
    save(fig, "fig_tradeoff")


def disagreement(res, metric, hb, op):
    agree = dis = 0
    for _, g in res.groupby("inst"):
        for (m1, o1), (m2, o2) in itertools.combinations(g[[metric, op]].to_numpy(), 2):
            dm, do = (m1 - m2) * (1 if hb else -1), o2 - o1
            if abs(dm) < 1e-12 or abs(do) < 1e-12:
                continue
            if (dm > 0) == (do > 0):
                agree += 1
            else:
                dis += 1
    return dis / max(1, agree + dis)


def fig_disagreement():
    metrics = [("pairF1", True, "pairwise F1"), ("B3F1", True, "B-cubed F1"), ("VI", False, "VI"),
               ("BMD", False, "BMD")]
    ops = [("COUNT_err", "COUNT"), ("SUM_err", "SUM"), ("CD_err", "COUNT\nDIST."),
           ("JOIN_FP_rate", "join\nfalse"), ("JOIN_FN_rate", "join\nmissed")]
    M = np.array([[disagreement(ea, m, hb, o) for o, _ in ops] for m, hb, _ in metrics])
    fig, ax = plt.subplots(figsize=(3.4, 2.0))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ops)), [o[1] for o in ops], fontsize=7)
    ax.set_yticks(range(len(metrics)), [m[2] for m in metrics], fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(M[i, j] - 0.5) > 0.3 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.05)
    cb.set_label("pairs ordered differently", fontsize=7)
    fig.tight_layout()
    save(fig, "fig_disagreement")


if __name__ == "__main__":
    fig_tradeoff()
    fig_disagreement()
    print("figures written to", OUT, sorted(os.listdir(OUT)))
