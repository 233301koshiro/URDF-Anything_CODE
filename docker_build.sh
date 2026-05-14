#!/bin/bash
# Build CPU-only Docker image for URDF-Anything
set -e
IMAGE_NAME=urdf_any_cpu:latest
CONTEXT_DIR=.

echo "Building Docker image: ${IMAGE_NAME}"
docker build -t ${IMAGE_NAME} ${CONTEXT_DIR}

echo "Built ${IMAGE_NAME}"