#!/usr/bin/env bash

set -x

EXP_DIR=detection_results/all-top1-row/
PY_ARGS=${@:1}

python -u detection-collect-all-top1.py \
    --limited_lesion --linear_eval \
    --output_dir ${EXP_DIR} --top_k_score 5 \
    ${PY_ARGS}
