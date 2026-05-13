#!/bin/bash
# Lightweight CPU evaluation: one sample per object ID (xxxxx_oo -> one xxxxx)

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$SCRIPT_DIR}"
IMAGE_NAME="${IMAGE_NAME:-urdf_any_cpu:latest}"

SRC_DATA_ROOT=${SRC_DATA_ROOT:-./datasets/urdf}
LIGHT_DATA_ROOT=${LIGHT_DATA_ROOT:-./datasets/urdf_light}
PREFERRED_MINOR=${PREFERRED_MINOR:-0}
LINK_MODE=${LINK_MODE:-symlink}
LIMIT_TEST_BATCHES=${LIMIT_TEST_BATCHES:--1}
WITH_PLY=${WITH_PLY:-false}
MAX_OBJECTS=${MAX_OBJECTS:-0}
OVERWRITE_LIGHT=${OVERWRITE_LIGHT:-false}

OUTPUT_DIR=${OUTPUT_DIR:-./output_light/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_test_light}
CKPT_PATH=${CKPT_PATH:-./checkpoints/last.ckpt}

if [ -z "${EVAL_LIGHT_IN_CONTAINER:-}" ] && [ ! -f "/.dockerenv" ]; then
  # Host mode: always execute inside the currently running urdf_any_cpu container.
  TARGET_CONTAINER=""

  EVAL_PID=$(ps -eo pid,args | awk '/python eval.py/ && $0 !~ /awk/ {print $1; exit}')
  if [ -n "$EVAL_PID" ] && [ -r "/proc/${EVAL_PID}/cgroup" ]; then
    CID_FROM_PID=$(grep -oE '[0-9a-f]{64}' "/proc/${EVAL_PID}/cgroup" | head -n 1 || true)
    if [ -n "$CID_FROM_PID" ] && docker inspect "$CID_FROM_PID" >/dev/null 2>&1; then
      TARGET_CONTAINER="$CID_FROM_PID"
    fi
  fi

  if [ -z "$TARGET_CONTAINER" ]; then
    TARGET_CONTAINER=$(docker ps --filter "ancestor=${IMAGE_NAME}" --format '{{.ID}}' | head -n 1)
  fi

  if [ -z "$TARGET_CONTAINER" ]; then
    echo "[eval_light] ERROR: running container not found for image ${IMAGE_NAME}" >&2
    echo "[eval_light] Start the container first, then run this script again." >&2
    exit 1
  fi

  CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-}"
  if [ -z "$CONTAINER_WORKSPACE" ]; then
    for cand in "/workspace/URDF-Anything" "/home/irsl/URDF-Anything_CODE_CPU"; do
      if docker exec "$TARGET_CONTAINER" bash -lc "[ -f '$cand/eval.py' ] && [ -f '$cand/eval_light.sh' ]" >/dev/null 2>&1; then
        CONTAINER_WORKSPACE="$cand"
        break
      fi
    done
  fi

  if [ -z "$CONTAINER_WORKSPACE" ]; then
    echo "[eval_light] ERROR: could not locate workspace path inside container ${TARGET_CONTAINER}" >&2
    echo "[eval_light] You can set CONTAINER_WORKSPACE and retry." >&2
    exit 1
  fi

  echo "[eval_light] Host mode -> container exec"
  echo "[eval_light] container: $TARGET_CONTAINER"
  echo "[eval_light] workspace in container: $CONTAINER_WORKSPACE"

  exec docker exec -i \
    -e EVAL_LIGHT_IN_CONTAINER=1 \
    -e SRC_DATA_ROOT="$SRC_DATA_ROOT" \
    -e LIGHT_DATA_ROOT="$LIGHT_DATA_ROOT" \
    -e PREFERRED_MINOR="$PREFERRED_MINOR" \
    -e LINK_MODE="$LINK_MODE" \
    -e LIMIT_TEST_BATCHES="$LIMIT_TEST_BATCHES" \
    -e WITH_PLY="$WITH_PLY" \
    -e MAX_OBJECTS="$MAX_OBJECTS" \
    -e OVERWRITE_LIGHT="$OVERWRITE_LIGHT" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e CKPT_PATH="$CKPT_PATH" \
    "$TARGET_CONTAINER" \
    bash -lc "cd '$CONTAINER_WORKSPACE' && ./eval_light.sh"
fi

LOG_DIR="${WORKSPACE}/logs"
mkdir -p "${LOG_DIR}"

# Compatibility patches (idempotent): keep behavior aligned with docker_run_eval.sh
sed -i 's/from pydantic.warnings import PydanticDeprecatedSince20/PydanticDeprecatedSince20 = Warning/g' train_lightning.py || true
sed -i 's/\*param.size()/param.size()/g' /usr/local/lib/python3.10/site-packages/transformers/modeling_utils.py || true
sed -i "s/assert self.precision in (16, 32), 'only 32 or 16 bit precision supported'/self.precision = 32/g" /usr/local/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py || true

echo "========================================================================="
echo "  LLaVA/LISA Lightweight Evaluation (CPU)"
echo "========================================================================="
echo "Source data: ${SRC_DATA_ROOT}"
echo "Light data:  ${LIGHT_DATA_ROOT}"
echo "Output dir:  ${OUTPUT_DIR}"
echo "Preferred minor: ${PREFERRED_MINOR}"
echo "Link mode: ${LINK_MODE}"
echo "Limit test batches: ${LIMIT_TEST_BATCHES}"
echo ""

BUILD_ARGS=(
  --src-root "${SRC_DATA_ROOT}"
  --dst-root "${LIGHT_DATA_ROOT}"
  --preferred-minor "${PREFERRED_MINOR}"
  --link-mode "${LINK_MODE}"
)

if [ "${WITH_PLY}" = "true" ]; then
  BUILD_ARGS+=(--with-ply)
fi
if [ "${OVERWRITE_LIGHT}" = "true" ]; then
  BUILD_ARGS+=(--overwrite)
fi
if [ "${MAX_OBJECTS}" != "0" ]; then
  BUILD_ARGS+=(--max-objects "${MAX_OBJECTS}")
fi

python ./scripts/build_light_eval_subset.py "${BUILD_ARGS[@]}"

mkdir -p "${OUTPUT_DIR}"
LOG_FILE="${LOG_DIR}/eval_light_$(date +%Y%m%d_%H%M%S).log"

python eval.py \
  --model_name_or_path ./checkpoints/ShapeLLM_7B_gapartnet_v1.0 \
  --version v1 \
  --vision_tower ./model/ReConV2/cfgs/pretrain/large/openshape.yaml \
  --vision_tower_path ./checkpoints/recon/large.pth \
  --backbone3d_path ./checkpoints/Uni3D/uni3d-b/model.pt \
  --data_root "${LIGHT_DATA_ROOT}" \
  --sample_points_num 2048 \
  --with_color True \
  --prompt_token_num 32 \
  --mm_use_pt_start_end False \
  --mm_use_pt_patch_token False \
  --lora_enable True \
  --lora_r 16 \
  --lora_alpha 32 \
  --output_dir "${OUTPUT_DIR}" \
  --per_device_eval_batch_size 1 \
  --model_max_length 2048 \
  --dataloader_num_workers 0 \
  --bf16 False \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --pretrain_mm_mlp_adapter ./checkpoints/mm_projector/mm_projector.bin \
  --load_ckpt_path "${CKPT_PATH}" \
  --limit_test_batches "${LIMIT_TEST_BATCHES}" \
  2>&1 | tee "${LOG_FILE}"

EVAL_EXIT=$?

echo ""
if [ ${EVAL_EXIT} -eq 0 ]; then
  echo "✓ Lightweight evaluation completed"
else
  echo "✗ Lightweight evaluation failed with exit code: ${EVAL_EXIT}"
fi
echo "Log: ${LOG_FILE}"
exit ${EVAL_EXIT}
