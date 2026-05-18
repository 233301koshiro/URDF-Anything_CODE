#!/bin/bash
# Custom dataset evaluation (CPU) for user-provided json/txt pairs

set -e
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WORKSPACE="${WORKSPACE:-$SCRIPT_DIR}"
IMAGE_NAME="${IMAGE_NAME:-urdf_any_cpu:latest}"

CUSTOM_DATA_ROOT="${CUSTOM_DATA_ROOT:-./datasets/rrbot_test}"
OUTPUT_DIR="${OUTPUT_DIR:-./output_custom/checkpoints/ShapeLLM_7B_gapartnet_v1.0-lora/infer_custom}"
CKPT_PATH="${CKPT_PATH:-./checkpoints/last.ckpt}"
LIMIT_TEST_BATCHES="${LIMIT_TEST_BATCHES:-1}"

if [ -z "${EVAL_CUSTOM_IN_CONTAINER:-}" ] && [ ! -f "/.dockerenv" ]; then
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
    echo "[eval_custom] ERROR: running container not found for image ${IMAGE_NAME}" >&2
    echo "[eval_custom] Start container first, then run this script again." >&2
    exit 1
  fi

  CONTAINER_WORKSPACE="${CONTAINER_WORKSPACE:-}"
  if [ -z "$CONTAINER_WORKSPACE" ]; then
    for cand in "/workspace/URDF-Anything" "/home/irsl/URDF-Anything_CODE_CPU"; do
      if docker exec "$TARGET_CONTAINER" bash -lc "[ -f '$cand/eval.py' ] && [ -f '$cand/eval_custom.sh' ]" >/dev/null 2>&1; then
        CONTAINER_WORKSPACE="$cand"
        break
      fi
    done
  fi

  if [ -z "$CONTAINER_WORKSPACE" ]; then
    echo "[eval_custom] ERROR: workspace path not found inside container ${TARGET_CONTAINER}" >&2
    echo "[eval_custom] Set CONTAINER_WORKSPACE and retry." >&2
    exit 1
  fi

  echo "[eval_custom] Host mode -> container exec"
  echo "[eval_custom] container: $TARGET_CONTAINER"
  echo "[eval_custom] workspace in container: $CONTAINER_WORKSPACE"

  exec docker exec -i \
    -e EVAL_CUSTOM_IN_CONTAINER=1 \
    -e CUSTOM_DATA_ROOT="$CUSTOM_DATA_ROOT" \
    -e OUTPUT_DIR="$OUTPUT_DIR" \
    -e CKPT_PATH="$CKPT_PATH" \
    -e LIMIT_TEST_BATCHES="$LIMIT_TEST_BATCHES" \
    "$TARGET_CONTAINER" \
    bash -lc "cd '$CONTAINER_WORKSPACE' && ./mine/scripts/eval/eval_custom.sh"
fi

LOG_DIR="${WORKSPACE}/logs"
mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_DIR}"

# Compatibility patches (idempotent)
sed -i 's/from pydantic.warnings import PydanticDeprecatedSince20/PydanticDeprecatedSince20 = Warning/g' train_lightning.py || true
sed -i 's/\*param.size()/param.size()/g' /usr/local/lib/python3.10/site-packages/transformers/modeling_utils.py || true
sed -i "s/assert self.precision in (16, 32), 'only 32 or 16 bit precision supported'/self.precision = 32/g" /usr/local/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py || true

if [ ! -d "${CUSTOM_DATA_ROOT}/json_questions" ] || [ ! -d "${CUSTOM_DATA_ROOT}/point_clouds" ]; then
  echo "[eval_custom] ERROR: dataset root must contain json_questions/ and point_clouds/" >&2
  echo "[eval_custom] got: ${CUSTOM_DATA_ROOT}" >&2
  exit 1
fi

PAIR_COUNT=$(python - <<'PY'
import os, glob
root = os.environ.get('CUSTOM_DATA_ROOT', './datasets/rrbot_test')
json_root = os.path.join(root, 'json_questions')
pc_root = os.path.join(root, 'point_clouds')
count = 0
for obj in sorted([d for d in os.listdir(json_root) if os.path.isdir(os.path.join(json_root,d))]):
    for jp in glob.glob(os.path.join(json_root, obj, '*.json')):
        base = os.path.splitext(os.path.basename(jp))[0]
        pp = os.path.join(pc_root, obj, base + '.txt')
        if os.path.exists(pp):
            count += 1
print(count)
PY
)

if [ "${PAIR_COUNT}" = "0" ]; then
  echo "[eval_custom] ERROR: no valid (json, txt) pairs found under ${CUSTOM_DATA_ROOT}" >&2
  exit 1
fi

echo "========================================================================="
echo "  LLaVA/LISA Custom Dataset Evaluation (CPU)"
echo "========================================================================="
echo "Data root: ${CUSTOM_DATA_ROOT}"
echo "Valid pairs: ${PAIR_COUNT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Checkpoint: ${CKPT_PATH}"
echo "Limit test batches: ${LIMIT_TEST_BATCHES}"
echo ""

LOG_FILE="${LOG_DIR}/eval_custom_$(date +%Y%m%d_%H%M%S).log"

python eval.py \
  --model_name_or_path ./checkpoints/ShapeLLM_7B_gapartnet_v1.0 \
  --version v1 \
  --vision_tower ./model/ReConV2/cfgs/pretrain/large/openshape.yaml \
  --vision_tower_path ./checkpoints/recon/large.pth \
  --backbone3d_path ./checkpoints/Uni3D/uni3d-b/model.pt \
  --data_root "${CUSTOM_DATA_ROOT}" \
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
  echo "✓ Custom dataset evaluation completed"
else
  echo "✗ Custom dataset evaluation failed with exit code: ${EVAL_EXIT}"
fi
echo "Log: ${LOG_FILE}"
exit ${EVAL_EXIT}
