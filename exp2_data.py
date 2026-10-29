"""E15 real-world dirty/clean datasets (exp2/), loaded as aligned string frames.

Layout: exp2/<name>/dirty.csv + clean.csv  (hospital: clean_long.csv, long format)
All values are read as strings with "" kept as a value.  Columns are aligned by
position because beers' dirty and clean headers differ in spelling
(beer_name vs beer-name); the clean header is used.
"""
import os
import pandas as pd

ROOT = os.environ.get("EXP2_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "exp2"))
ROW_ID = {"hospital": None, "flights": "tuple_id", "beers": "index", "rayyan": None}


def load(name):
    d = os.path.join(ROOT, name)
    read = lambda p: pd.read_csv(p, dtype=str, keep_default_na=False)
    dirty = read(os.path.join(d, "dirty.csv"))
    if name == "hospital":
        long = read(os.path.join(d, "clean_long.csv"))
        clean = long.pivot(index="tid", columns="attribute", values="correct_val")
        clean.index = clean.index.astype(int)
        clean = clean.sort_index().reset_index(drop=True).reindex(columns=dirty.columns)
    else:
        clean = read(os.path.join(d, "clean.csv"))
        assert clean.shape == dirty.shape, (name, clean.shape, dirty.shape)
        dirty.columns = clean.columns
    rid = ROW_ID[name]
    if rid:
        assert (dirty[rid] == clean[rid]).all(), f"{name}: row ids not aligned"
    return dirty.reset_index(drop=True), clean.reset_index(drop=True)


def group_violation(df, X, y):
    """Fraction of X-groups with more than one distinct y (our violation rate)."""
    return float((df.groupby(list(X))[y].nunique() > 1).mean())


def select_candidates(clean, exclude=(), n_true=10, n_false=10, seed=0, false_min=0.5):
    """Pre-registered rule: single-attribute candidates X -> Y over the clean
    table.  True  = violation 0, X not a row key, Y not constant.
              False = violation >= false_min.  0 < violation < false_min is
    ambiguous and excluded.  Sample n_true / n_false with a fixed seed."""
    import random
    cols = [c for c in clean.columns if c not in exclude]
    n = len(clean)
    true, false, ambiguous = [], [], 0
    for x in cols:
        if clean[x].nunique() in (1, n):
            continue
        for y in cols:
            if y == x or clean[y].nunique() == 1:
                continue
            v = group_violation(clean, [x], y)
            if v == 0:
                true.append((x, y))
            elif v >= false_min:
                false.append((x, y))
            else:
                ambiguous += 1
    rng = random.Random(seed)
    pick_t = sorted(rng.sample(sorted(true), min(n_true, len(true))))
    pick_f = sorted(rng.sample(sorted(false), min(n_false, len(false))))
    return ([(x, y, True) for x, y in pick_t] + [(x, y, False) for x, y in pick_f],
            {"true_pool": len(true), "false_pool": len(false), "ambiguous": ambiguous})
