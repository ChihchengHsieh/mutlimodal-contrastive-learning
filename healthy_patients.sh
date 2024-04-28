#!/usr/bin/env bash

set -x

EXP_DIR=detection_results/healthy/
PY_ARGS=${@:1}

python -u healthy_patients.py \
    --linear_eval \
    --output_dir ${EXP_DIR} \
    ${PY_ARGS}
