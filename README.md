# VetoQ: Operator-Aware Entity Equivalence for Semantic Queries over Dirty Keys

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23756806.svg)](https://doi.org/10.5281/zenodo.23756806)

Code, data and experiment logs for the paper

> **Merges Are Not Splits: Operator-Aware Entity Equivalence for Semantic Queries over Dirty Keys**
> Mukhiddin Toshpulatov (Gachon University), Wookey Lee (Inha University), Suan Lee (Semyung University).
> Submitted to ICDE 2027.

Semantic query operators group and join dirty keys by similarity. The induced equivalence errs in two ways:
a *False-Merge* puts distinct entities into one class, and a *Split* spreads one entity over several classes.
The paper derives each operator's error from the class–entity contingency table, proves that transitive
closure amplifies merges but not splits, and presents VetoQ, which chooses the equivalence per operator.
Full proofs are in [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md).

## Contents

| Path | What it is |
|---|---|
| `exp_t1/` | Experiment code. `contingency.py` (error calculus, tie rules, chooser), `grouping.py` (the 24 candidate equivalences), `hard_multi.py`, `datasets.py`, `run_e*.py`, `analyze_*.py`, `make_figures.py`, `run_all.sh` |
| `hard_bench.py`, `run_first.py`, `embedding_backend.py`, `exp2_data.py` | Modules the experiments import: the Hard-2000 generator, similarities, sentence embeddings, real-data loader |
| `exp2/`, `data_t1/` | Real datasets (see [Data](#data)) |
| `gpu_out/t1_*` | Logs behind every number in the paper |
| `gpu_out/v1.0_2026-10-23/` | Logs of the first run, the source of the run times quoted in the paper |
| `e6_*.txt` | Embedding-similarity logs quoted in §I |
| `figures_t1/` | Figures 1 and 2 (output of `make_figures.py`) |

## Setup

Python 3.14 with [uv](https://docs.astral.sh/uv/):

```bash
bash setup_env.sh        # creates .venv; torch is the CUDA 12.6 build, but every experiment runs on CPU
```

`requirements.txt` pins the direct dependencies (pandas 3.0.6, numpy 2.4.3, scikit-learn 1.9.0,
sentence-transformers 6.1.0, transformers 5.19.0, torch 2.10.0, splink 5.0.0, duckdb 2.0.0);
`requirements-lock.txt` is the full Linux freeze. A CPU-only torch build (`--index-url
https://download.pytorch.org/whl/cpu`) gives the same results. On Windows, run the scripts from Git Bash;
`run_all.sh` finds `.venv/Scripts/python.exe` automatically.

## Reproducing the results

The scripts write into `gpu_out/` and would overwrite the shipped logs, so reproduce them in a working copy:

```bash
bash exp_t1/run_all.sh                          # all experiments, ~45 min on 32 CPU threads
ONLY="ea ed eh" bash exp_t1/run_all.sh          # a subset: ec ea eb ed ednoise eer eex ehc eg eh ef fig
```

`run_all.sh` copies the repository to `../VetoQ_repro_<date>`, runs there with `CUDA_VISIBLE_DEVICES=""`,
and prints `IDENTICAL` or `DIFFERS` for every log compared with the shipped one. To run one script by hand:
`CUDA_VISIBLE_DEVICES="" .venv/bin/python -P exp_t1/run_ea.py` (`run_ec_ea.py` runs without `-P`).
Run `run_ea.py` first: the E-G, E-H and figure scripts read its `t1_ea.csv`.

| Experiment | Paper | Script | Log |
|---|---|---|---|
| E-C exactness | §VII | `run_ec_ea.py` | `t1_ec_ea.txt` |
| E-A metrics vs. operator errors | Fig. 1, Tables V, VII | `run_ea.py` | `t1_ea.txt`, `t1_ea.csv` |
| E-B closure asymmetry | Table VI | `run_eb.py` | `t1_eb.txt` |
| E-D pipelines and choosers (Hard-multi) | Tables VIII, IX | `run_ed.py` (`--wit-noise 0.1` for the noisy witness); `check_ed_mixed.py` for the mixed mass of the Context Veto candidates (§VII-D) | `t1_ed*.txt` |
| E-E real data (Hospital), Splink candidates | Table IX | `run_ee_real.py`, `run_ee_ext.py` | `t1_ee_*.txt` |
| E-G LLM-based grouping | Table X | `run_eg.py`, `analyze_eg.py` | `t1_eg_*.txt` |
| E-H Splink | Tables XI, XII | `run_eh.py`, `analyze_eh.py`, `run_eh_closure.py` | `t1_eh*.txt` |
| E-F scale (MusicBrainz 20K/200K) | Table XIII | `run_ef.py` | `t1_ef.txt` |

Section H of the technical report maps every table to its script and log in more detail.

Notes on reproducibility:

- All errors in the paper come from CPU runs. A rerun on a different CPU reproduces them, except for a few
  ties at the ε boundary of complete linkage and in nearest-neighbour search (some E-A and E-B Hard-2000
  cells, and the E-F 200K BMD in the last digit); no number quoted in the paper changes.
- E-G (Qwen2.5-1.5B-Instruct) reads its answers from `gpu_out/t1_eg_cache_hf_Qwen2.5-1.5B-Instruct.json` and
  finishes in seconds; without the cache it takes about 4.7 CPU-hours.
- Quoted run times come from the first run on an AMD Ryzen 7 3700X (`gpu_out/v1.0_2026-10-23/`).
- E-C compares eight groupings of Hard-2000, including three Context Veto forms selected by
  `hard_bench.canon(..., ctx_mode=...)`: `"hard"` (minimum of value and context similarity), `"soft"`
  (context similarity ≥ 0.5 counts as a match) and `"canon"` (context values clustered by complete linkage at
  distance ε; two contexts match iff they share a cluster). Version 1.0.0 lacked `ctx_mode`, so
  `run_ec_ea.py` failed; from 1.0.1 it reproduces `gpu_out/t1_ec_ea.txt` byte for byte.
- Version 1.0.2 adds `exp_t1/check_ed_mixed.py` (run as `edx` by `run_all.sh`), which writes
  `gpu_out/t1_ed_mixed.txt`: the hard Context Veto has mixed mass X = 0 (collapse-free) on all 15 Hard-multi
  instances under both linkages.

## Data

| Dataset | Path | Source | Terms |
|---|---|---|---|
| Hard-2000, Hard-multi | generated by `hard_bench.py`, `exp_t1/hard_multi.py` | this work | CC BY 4.0 |
| Hospital, Beers | `exp2/` | Raha benchmark (M. Mahdavi et al., "Raha: A configuration-free error detection system", SIGMOD 2019), [github.com/BigDaMa/raha](https://github.com/BigDaMa/raha) | Apache License 2.0 ([`exp2/LICENSE-raha.md`](exp2/LICENSE-raha.md)) |
| MusicBrainz 20K/200K, Geographic Settlements, Affiliations | `data_t1/` | Leipzig benchmark datasets for entity resolution (A. Saeedi, E. Peukert, E. Rahm, ADBIS 2017), [dbs.uni-leipzig.de](https://dbs.uni-leipzig.de/research/projects/benchmark-datasets-for-entity-resolution) | CC BY 4.0 |

The third-party datasets are redistributed unchanged for reproducibility; please cite their original sources.

## Citation

See [`CITATION.cff`](CITATION.cff). Until the paper is published, please cite this repository:

> M. Toshpulatov, W. Lee, and S. Lee, "Merges are not splits: Technical report, code, data and logs," 2026,
> doi: [10.5281/zenodo.23756806](https://doi.org/10.5281/zenodo.23756806).

## License

Code: [MIT](LICENSE). Data generated by this work and the experiment logs: CC BY 4.0.
Third-party datasets keep their original terms (see [Data](#data)).
