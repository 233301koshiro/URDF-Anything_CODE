#!/bin/bash
# Build image if needed, then run test script inside container and stream logs
set -e
IMAGE_NAME=urdf_any_cpu:latest
WORKDIR_HOST=$(pwd)
CONTAINER_WORKDIR=/workspace/URDF-Anything
LOG_DIR=${WORKDIR_HOST}/logs
mkdir -p "$LOG_DIR"

# If image not present, build it
if ! docker image inspect ${IMAGE_NAME} > /dev/null 2>&1; then
  echo "Image ${IMAGE_NAME} not found. Building first..."
  bash docker_build.sh
fi

# Run the test script inside container (non-interactive) and tee logs
docker run --rm \
  -v "${WORKDIR_HOST}:${CONTAINER_WORKDIR}" \
  -w "${CONTAINER_WORKDIR}" \
  ${IMAGE_NAME} \
  bash -lc "python test_llava_cpu.py" 2>&1 | tee "${LOG_DIR}/docker_test_$(date +%Y%m%d_%H%M%S).log"

