"""HANDOFF 6.C-4: mixed mass X of the Context Veto candidates on Hard-multi (same 15 instances as run_ed.py).
X = 0 means collapse-free (every class meets one entity); reported on the union of all tables and on orders.
Writes gpu_out/t1_ed_mixed.txt. Usage: python -P exp_t1/check_ed_mixed.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import contingency as C          # noqa: E402
import grouping as G             # noqa: E402
import run_ed as R               # noqa: E402
from hard_multi import generate  # noqa: E402

lines = ["seed rate method X_union X_orders"]
for seed in range(5):
    for rate in (0.1, 0.2, 0.3):
        countries, customers, orders = generate(seed, rate)
        u = R.union_frame(countries, customers, orders)
        truth = u["E"].astype(str).to_numpy()
        O = (u["tbl"] == "O").to_numpy()
        G.clear_cache()
        for name, kw in G.method_grid("str").items():
            if "veto" not in name:
                continue
            lab = np.asarray(G.group(u, ctx_kind="str", **kw)).astype(str)
            xu = C.operator_errors(C.table(lab, truth))["X"]
            xo = C.operator_errors(C.table(lab[O], truth[O]))["X"]
            lines.append(f"{seed} {rate} {name} {xu} {xo}")
out = os.path.join(HERE, "..", "gpu_out", "t1_ed_mixed.txt")
with open(out, "w", encoding="utf-8", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
