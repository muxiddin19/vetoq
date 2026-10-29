"""Hard-2000 multi-table generator for pipeline experiments (E-D).

Tables (one random.Random(seed) drives everything, so each (seed, rate) is reproducible):
  countries(name, capital, continent, code)          clean dimension, 12 rows (hard_bench.COUNTRIES)
  customers(cust_id, country, capital, code, E)      n_cust rows; country/capital dirty, E = true country
  orders(order_id, cust_id, country, capital, amount, code, E)
                                                     n_orders rows; the order re-types the country and capital
                                                     (dirty, independently of the customer row); E = true country
  code: the calling code, an entity-constant identifier used only as a (partially observed) witness in E-E;
        with wit_noise > 0, each recorded code is replaced by another entity's code with that probability
Country popularity is skewed (Zipf-like), so look-alike merges can join a large and a small entity.
Noise follows hard_bench.corrupt: per cell, with probability `rate`, a synonym (p=1/2 when one exists) or a typo.
"""
import os
import random
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hard_bench import COUNTRIES, SYNONYMS, _typo      # noqa: E402


def _noisy(rng, v, rate):
    if rng.random() < rate:
        if v in SYNONYMS and rng.random() < 0.5:
            return rng.choice(SYNONYMS[v])
        return _typo(rng, v)
    return v


def _wit(rng, code, codes, wit_noise):
    """Witness value as recorded: with probability wit_noise a wrong code (another entity's), else the true one."""
    if wit_noise and rng.random() < wit_noise:
        return rng.choice([c for c in codes if c != code])
    return code


def generate(seed, rate, n_cust=400, n_orders=5000, zipf=1.1, wit_noise=0.0):
    rng = random.Random(seed)
    wrng = random.Random(seed * 7919 + 1)                # separate stream: wit_noise=0 reproduces the data exactly
    names = list(COUNTRIES)
    order = names[:]
    rng.shuffle(order)                                   # random popularity ranks per seed
    weights = [1.0 / (i + 1) ** zipf for i in range(len(order))]
    countries = pd.DataFrame([{"name": c, "capital": COUNTRIES[c][0], "continent": COUNTRIES[c][1],
                               "code": COUNTRIES[c][3]} for c in names])
    codes = [COUNTRIES[c][3] for c in names]
    cust = []
    for i in range(n_cust):
        c = rng.choices(order, weights)[0]
        cust.append({"cust_id": i, "country": _noisy(rng, c, rate),
                     "capital": _noisy(rng, COUNTRIES[c][0], rate),
                     "code": _wit(wrng, COUNTRIES[c][3], codes, wit_noise), "E": c})
    customers = pd.DataFrame(cust)
    ords = []
    for j in range(n_orders):
        k = rng.randrange(n_cust)
        c = cust[k]["E"]
        ords.append({"order_id": j, "cust_id": k, "country": _noisy(rng, c, rate),
                     "capital": _noisy(rng, COUNTRIES[c][0], rate),
                     "amount": round(rng.lognormvariate(3.0, 1.0), 2),
                     "code": _wit(wrng, COUNTRIES[c][3], codes, wit_noise), "E": c})
    orders = pd.DataFrame(ords)
    return countries, customers, orders


if __name__ == "__main__":
    co, cu, od = generate(0, 0.2)
    print(co.to_string(index=False))
    print(cu.head().to_string(index=False))
    print(od.head().to_string(index=False))
    print("orders per true country:\n", od["E"].value_counts().to_string())
    print("distinct dirty country spellings in orders:", od["country"].nunique())
