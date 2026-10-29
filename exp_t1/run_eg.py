"""E-G: LLM-based grouping (canonicalize, then GROUP BY) versus VetoQ's candidates.

Each distinct (key, context) combination is mapped by an LLM to a canonical entity name, as a semantic map
followed by an exact GROUP BY (the LOTUS sem_map pattern). Two variants: value only, and value + context.
Classes = normalized LLM answers. Scored with exactly the metrics and random draws of run_ea.py, so the numbers
are comparable with gpu_out/t1_ea.csv instance by instance.

Backends
  hf      local instruct model on CPU via transformers (default Qwen/Qwen2.5-1.5B-Instruct)
  claude  Anthropic API (needs credentials; default model claude-opus-5-5), 50 values per request, JSON reply
Answers are cached in gpu_out/t1_eg_cache_<backend>_<model>.json, so runs can resume.

Usage: python -P exp_t1/run_eg.py [--backend hf] [--model Qwen/Qwen2.5-1.5B-Instruct] [--batch 16]
"""
import argparse
import json
import os
import re
import sys
import time
import zlib

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import contingency as C          # noqa: E402
import datasets as D             # noqa: E402

KIND = {"Hard-2000": ("country", "capital"), "Hospital": ("hospital", "state"), "Beers": ("brewery", "city"),
        "MusicBrainz": ("song title", "artist"), "Settlements": ("city or settlement", "coordinates (lat, lon)"),
        "Affiliations": ("research institution or company", None)}
SYSTEM = ("You normalize dirty database values to canonical entity names. Different spellings, typos, "
          "abbreviations and synonyms of the same real-world entity must get the same canonical name; "
          "different entities must get different names, even if their names look alike. "
          "Reply with the canonical name only, in English, without explanation.")


def instances():
    out = []
    for s in range(5):
        for r in (0.1, 0.2, 0.3):
            out.append(lambda s=s, r=r: D.hard(s, r))
    out += [D.hospital, D.beers, lambda: D.musicbrainz(0, 1000), lambda: D.settlements(0, 400), D.affiliations]
    return out


def ctx_strings(df, meta):
    if meta["ctx_kind"] == "geo":
        return [("" if np.isnan(a) else f"{a:.3f}, {b:.3f}") for a, b in zip(df["LAT"], df["LON"])]
    if meta["ctx_kind"] == "str":
        return [str(c) for c in df["CTX"]]
    return [""] * len(df)


def user_prompt(kind, ctx_label, value, ctx):
    s = f"Entity type: {kind}.\nValue: {value}\n"
    if ctx_label and ctx:
        s += f"{ctx_label}: {ctx}\n"
    return s + "Canonical name:"


def canon_answer(a):
    a = a.strip().split("\n")[0]
    a = re.sub(r"[\"'`*.,;:()\[\]]", " ", a.lower())
    return " ".join(a.split())


class HFBackend:
    def __init__(self, model, batch):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(0)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.tok.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32)
        self.model.eval()
        self.batch = batch
        self.torch = torch

    def run(self, prompts):
        out, n_in, n_out = [], 0, 0
        for i in range(0, len(prompts), self.batch):
            chunk = prompts[i:i + self.batch]
            texts = [self.tok.apply_chat_template([{"role": "system", "content": SYSTEM},
                                                   {"role": "user", "content": p}],
                                                  tokenize=False, add_generation_prompt=True) for p in chunk]
            enc = self.tok(texts, return_tensors="pt", padding=True)
            with self.torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=32, do_sample=False,
                                          pad_token_id=self.tok.eos_token_id)
            new = gen[:, enc["input_ids"].shape[1]:]
            out += self.tok.batch_decode(new, skip_special_tokens=True)
            n_in += int(enc["attention_mask"].sum())
            n_out += int((new != self.tok.eos_token_id).sum())
        return out, n_in, n_out


class ClaudeBackend:
    """50 values per request; the model returns a JSON list of canonical names in input order."""

    def __init__(self, model):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def run(self, prompts):
        out, n_in, n_out = [], 0, 0
        for i in range(0, len(prompts), 50):
            chunk = prompts[i:i + 50]
            body = "\n\n".join(f"[{j}]\n{p}" for j, p in enumerate(chunk))
            resp = self.client.beta.messages.create(
                model=self.model, max_tokens=16000, system=SYSTEM,
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": body + f"\n\nReturn a JSON array of exactly {len(chunk)} "
                                                                "strings, the canonical name for [0], [1], ... "
                                                                "in order, and nothing else."}])
            if resp.stop_reason == "refusal":
                raise RuntimeError(f"refused: {resp.stop_details}")
            text = next(b.text for b in resp.content if b.type == "text")
            names = json.loads(text[text.find("["): text.rfind("]") + 1])
            if len(names) != len(chunk):
                raise RuntimeError(f"expected {len(chunk)} names, got {len(names)}")
            out += [str(x) for x in names]
            n_in += resp.usage.input_tokens
            n_out += resp.usage.output_tokens
        return out, n_in, n_out


def score(df, inst_id, lab):
    """Same metrics and random draws as run_ea.evaluate."""
    truth = df["E"].astype(str).to_numpy()
    rng = np.random.default_rng(zlib.crc32(inst_id.encode()))
    amount = rng.lognormal(3.0, 1.0, len(df)).round(2)
    side = rng.random(len(df)) < 0.5
    true_sum = pd.Series(amount).groupby(truth).sum()
    lab = np.asarray(lab).astype(str)
    n = C.table(lab, truth)
    ms = C.masses(n)
    er = C.er_metrics(n)
    op = C.operator_errors(n)
    n_w = pd.crosstab(pd.Series(lab, name="c"), pd.Series(truth, name="e"), values=amount,
                      aggfunc="sum").fillna(0.0).reindex(index=n.index, columns=n.columns)
    f_sum = C.formula_groupby_sum(n_w, ms)
    j = C.formula_join(C.table(lab[side], truth[side]), C.table(lab[~side], truth[~side]))
    jp = j["TP"] / (j["TP"] + j["FP"]) if j["TP"] + j["FP"] else 1.0
    jr = j["TP"] / (j["TP"] + j["FN"]) if j["TP"] + j["FN"] else 1.0
    return dict(**er, **op, SUM_err=float((f_sum.abs() / true_sum.reindex(f_sum.index)).mean()),
                JOIN_FP_rate=1 - jp, JOIN_FN_rate=1 - jr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["hf", "claude"], default="hf")
    ap.add_argument("--model", default=None)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    model = args.model or ("Qwen/Qwen2.5-1.5B-Instruct" if args.backend == "hf" else "claude-opus-5-5")
    tag = f"{args.backend}_{model.split('/')[-1]}"
    cache_path = os.path.join(ROOT, "gpu_out", f"t1_eg_cache_{tag}.json")
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    stats = cache.pop("__stats__", {"calls_prompts": 0, "tok_in": 0, "tok_out": 0, "sec": 0.0})

    loaded = []
    need = {}
    for loader in instances():
        df, meta = loader()
        ds = meta["name"].split(" ")[0]
        kind, ctx_label = KIND[ds]
        ctx = ctx_strings(df, meta)
        loaded.append((df, meta, ds, ctx))
        for variant in (["value"] + (["context"] if ctx_label else [])):
            for k, c in zip(df["K"].astype(str), ctx):
                key = json.dumps([ds, variant, k, c if variant == "context" else ""], ensure_ascii=False)
                if key not in cache:
                    need[key] = user_prompt(kind, ctx_label if variant == "context" else None, k,
                                            c if variant == "context" else "")
    print(f"[E-G] backend={tag} prompts to run: {len(need)} (cached: {len(cache)})", flush=True)
    if need:
        be = HFBackend(model, args.batch) if args.backend == "hf" else ClaudeBackend(model)
        keys = list(need)
        step = 256
        for i in range(0, len(keys), step):
            t0 = time.perf_counter()
            ans, n_in, n_out = be.run([need[k] for k in keys[i:i + step]])
            stats["sec"] += time.perf_counter() - t0
            stats["calls_prompts"] += len(ans)
            stats["tok_in"] += n_in
            stats["tok_out"] += n_out
            cache.update(dict(zip(keys[i:i + step], ans)))
            json.dump({**cache, "__stats__": stats}, open(cache_path, "w", encoding="utf-8"), ensure_ascii=False)
            print(f"  {min(i + step, len(keys))}/{len(keys)} prompts, {stats['sec']:.0f}s", flush=True)

    rows = []
    for df, meta, ds, ctx in loaded:
        for variant in (["value"] + (["context"] if KIND[ds][1] else [])):
            lab = [canon_answer(cache[json.dumps([ds, variant, k, c if variant == "context" else ""],
                                                 ensure_ascii=False)])
                   for k, c in zip(df["K"].astype(str), ctx)]
            rows.append(dict(dataset=ds, inst=meta["name"], method=f"LLM-{variant}", **score(df, meta["name"], lab)))
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ROOT, "gpu_out", f"t1_eg_{tag}.csv"), index=False)

    ea = pd.read_csv(os.path.join(ROOT, "gpu_out", "t1_ea.csv"))
    ops = ["COUNT_err", "SUM_err", "CD_err", "JOIN_FP_rate", "JOIN_FN_rate"]
    comp = []
    for ds, g in res.groupby("dataset"):
        e = ea[(ea.dataset == ds) & ea.inst.isin(g.inst)]
        line = {"dataset": ds}
        for m in ("LLM-value", "LLM-context"):
            if (g.method == m).any():
                for o in ops:
                    line[f"{m}:{o}"] = g.loc[g.method == m, o].mean()
        for name, sel in (("hardveto", e.method == "hybrid-any-0.2+hardveto"), ("norm", e.method == "norm")):
            if sel.any():
                for o in ops:
                    line[f"{name}:{o}"] = e.loc[sel, o].mean()
        for o in ops:
            line[f"best:{o}"] = e.groupby("inst")[o].min().mean()
        comp.append(line)
    comp = pd.DataFrame(comp).set_index("dataset")
    print(f"\n=== E-G ({tag}): mean operator errors per dataset; best = best of 24 VetoQ candidates per instance ===")
    for o in ops:
        cols = [c for c in comp.columns if c.endswith(":" + o)]
        print(f"\n[{o}]")
        print(comp[cols].rename(columns=lambda c: c.split(":")[0]).to_string(float_format=lambda v: f"{v:.3f}"))
    print(f"\nLLM cost: {stats['calls_prompts']} prompts, {stats['tok_in']:,} input / {stats['tok_out']:,} output "
          f"tokens, {stats['sec']:.0f} s")


if __name__ == "__main__":
    main()

