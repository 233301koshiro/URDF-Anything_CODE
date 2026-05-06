#!/bin/bash
# LLaVA/LISA Evaluation (eval.sh) を実行するコンテナ起動スクリプト
set -e
IMAGE_NAME=urdf_any_cpu:latest
WORKDIR_HOST=$(pwd)
CONTAINER_WORKDIR=/workspace/URDF-Anything
LOG_DIR=${WORKDIR_HOST}/logs
mkdir -p "$LOG_DIR"

echo "Starting evaluation container..."

# Install dependencies and run evaluation
docker run --rm \
  -v "${WORKDIR_HOST}:${CONTAINER_WORKDIR}" \
  -w "${CONTAINER_WORKDIR}" \
  ${IMAGE_NAME} \
  bash -lc "pip install open3d 'setuptools<70.0.0' 'pydantic<2.0' 'transformers==4.31.0' && \
  sed -i 's/from pydantic.warnings import PydanticDeprecatedSince20/PydanticDeprecatedSince20 = Warning/g' train_lightning.py && \
  sed -i 's/\*param.size()/param.size()/g' /usr/local/lib/python3.10/site-packages/transformers/modeling_utils.py && \
  sed -i \"s/assert self.precision in (16, 32), 'only 32 or 16 bit precision supported'/self.precision = 32/g\" /usr/local/lib/python3.10/site-packages/pytorch_lightning/trainer/trainer.py && \
  bash ./eval.sh" 2>&1 | tee "${LOG_DIR}/docker_eval_$(date +%Y%m%d_%H%M%S).log"