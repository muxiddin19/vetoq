#!/usr/bin/env bash
# Reproduce every T1 log in a working copy and compare it with gpu_out/ (the scripts write into <root>/gpu_out,
# so running them in place would overwrite the logs the paper cites).
#
# Usage: bash exp_t1/run_all.sh [WORKDIR]            (default: ../VetoQ_repro_<date>)
#        ONLY="ec ed" bash exp_t1/run_all.sh          run a subset: ec ea eb ed ednoise eer eex ehc eg eh ef fig
# Runtime on 32 CPU threads: about 45 min in total (E-A 12-27 min, E-F 23 min, the rest 1-6 min each).
# E-G reads the answer cache gpu_out/t1_eg_cache_hf_Qwen2.5-1.5B-Instruct.json; without it, it needs ~4.7 CPU-hours.
# Numbers in the paper are CPU runs: CUDA_VISIBLE_DEVICES="" is set below.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${1:-$ROOT/../VetoQ_repro_$(date +%Y%m%d_%H%M)}"
PY="$ROOT/.venv/bin/python"
[[ -x $PY ]] || PY="$ROOT/.venv/Scripts/python.exe"   # uv venv on Windows (Git Bash)
ONLY="${ONLY:-ec ea eb ed ednoise eer eex ehc eg eh ef fig}"
export CUDA_VISIBLE_DEVICES="" HF_HUB_DISABLE_PROGRESS_BARS=1 PYTHONUTF8=1

mkdir -p "$WORK"
cp -a "$ROOT"/*.py "$ROOT"/exp_t1 "$ROOT"/exp2 "$ROOT"/data_t1 "$ROOT"/gpu_out "$ROOT"/e6_*.txt "$WORK"/
mkdir -p "$WORK/figures_t1" "$WORK/logs"
cd "$WORK"
echo "working copy: $WORK"

run() {  # run <tag> <stdout log under gpu_out/ or -> <script> [args...]
  local tag=$1 out=$2; shift 2
  [[ " $ONLY " == *" $tag "* ]] || return 0
  local t0=$SECONDS
  if [[ $out == - ]]; then "$PY" "$@" > "logs/$tag.out" 2> "logs/$tag.err"
  else "$PY" "$@" > "gpu_out/$out" 2> "logs/$tag.err"; fi
  echo "[$tag] rc=$? $((SECONDS - t0))s  $*"
}
# order matters: run_ea writes t1_ea.csv, which run_eg, run_eh, analyze_* and make_figures read
run ec      t1_ec_ea.txt             exp_t1/run_ec_ea.py            # imports its sibling module: no -P
run ea      t1_ea.txt                -P exp_t1/run_ea.py
run eb      t1_eb.txt                -P exp_t1/run_eb.py
run ed      t1_ed.txt                -P exp_t1/run_ed.py
run ednoise t1_ed_witnoise0.1.txt    -P exp_t1/run_ed.py --wit-noise 0.1 --tag _witnoise0.1
run eer     t1_ee_hospital.txt       -P exp_t1/run_ee_real.py
run eex     t1_ee_ext.txt            -P exp_t1/run_ee_ext.py
run ehc     t1_eh_closure.txt        -P exp_t1/run_eh_closure.py
run eg      t1_eg_hf.txt             -P exp_t1/run_eg.py --backend hf --batch 32
run eg      t1_eg_analysis.txt       -P exp_t1/analyze_eg.py
run eh      t1_eh.txt                -P exp_t1/run_eh.py
run eh      t1_eh_analysis.txt       -P exp_t1/analyze_eh.py
run ef      t1_ef.txt                -P exp_t1/run_ef.py
run fig     -                        -P exp_t1/make_figures.py

echo; echo "=== comparison with $ROOT/gpu_out (CR/BOM stripped; timing columns differ by machine) ==="
for f in gpu_out/t1_*.txt gpu_out/t1_*.csv; do
  o="$ROOT/$f"; [[ -f $o ]] || { echo "NEW        $f"; continue; }
  if diff -q <(sed '1s/^\xEF\xBB\xBF//' "$o" | tr -d '\r') <(sed '1s/^\xEF\xBB\xBF//' "$f" | tr -d '\r') > /dev/null
  then echo "IDENTICAL  $f"; else echo "DIFFERS    $f"; fi
done
