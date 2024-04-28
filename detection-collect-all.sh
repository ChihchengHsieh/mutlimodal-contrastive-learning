#!/usr/bin/env bash

set -x

EXP_DIR=detection_results/all/
PY_ARGS=${@:1}

python -u detection-collect-all.py \
    --limited_lesion --linear_eval \
    --output_dir ${EXP_DIR} --top_k_score 5 \
    ${PY_ARGS}
