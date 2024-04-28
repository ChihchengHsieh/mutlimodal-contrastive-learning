#!/usr/bin/env bash

set -x

PY_ARGS=${@:1}

python -u detection-collect-top5.py \
    --limited_lesion --linear_eval \
    ${PY_ARGS}
