#!/bin/bash
# Quick lightweight benchmark: one sample per object ID, capped object count

set -e

MAX_OBJECTS=${MAX_OBJECTS:-50}
PREFERRED_MINOR=${PREFERRED_MINOR:-0}
LIMIT_TEST_BATCHES=${LIMIT_TEST_BATCHES:-1}
OVERWRITE_LIGHT=${OVERWRITE_LIGHT:-true}
LINK_MODE=${LINK_MODE:-symlink}

OUTPUT_DIR=${OUTPUT_DIR:-./output_light/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test_light_quick}
LIGHT_DATA_ROOT=${LIGHT_DATA_ROOT:-./datasets/urdf_light_quick}

export MAX_OBJECTS
export PREFERRED_MINOR
export LIMIT_TEST_BATCHES
export OVERWRITE_LIGHT
export LINK_MODE
export OUTPUT_DIR
export LIGHT_DATA_ROOT

echo "[quick] MAX_OBJECTS=${MAX_OBJECTS}"
echo "[quick] PREFERRED_MINOR=${PREFERRED_MINOR}"
echo "[quick] LIMIT_TEST_BATCHES=${LIMIT_TEST_BATCHES}"
echo "[quick] LIGHT_DATA_ROOT=${LIGHT_DATA_ROOT}"
echo "[quick] OUTPUT_DIR=${OUTPUT_DIR}"

exec ./eval_light.sh
