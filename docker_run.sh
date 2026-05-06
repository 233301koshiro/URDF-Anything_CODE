#!/bin/bash
# Run an interactive container for URDF-Anything (CPU)
set -e
IMAGE_NAME=urdf_any_cpu:latest
WORKDIR_HOST=$(pwd)
CONTAINER_WORKDIR=/workspace/URDF-Anything

# Mount the project into container to see logs and allow edits
# Map local logs/ and checkpoints/ if present

docker run --rm -it \
  -v "${WORKDIR_HOST}:${CONTAINER_WORKDIR}" \
  -w "${CONTAINER_WORKDIR}" \
  --name urdf_any_cpu_run \
  ${IMAGE_NAME} bash
