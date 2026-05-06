#!/bin/bash
# LLaVA CPU Test Script with Progress Monitoring
# Shows real-time progress and execution logs

set -e

WORKSPACE="/home/irsl/URDF-Anything_CODE_CPU"
LOG_DIR="${WORKSPACE}/logs"

# Create log directory
mkdir -p "${LOG_DIR}"

echo "========================================================================="
echo "  LLaVA CPU Verification Test"
echo "========================================================================="
echo ""
echo "Workspace: ${WORKSPACE}"
echo "Log directory: ${LOG_DIR}"
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "✗ Error: python3 not found"
    exit 1
fi

echo "Python version: $(python3 --version)"
echo ""

# Run the test with output to both console and log file
echo "Starting test... (output saved to test.log)"
echo ""

LOG_FILE="${LOG_DIR}/test.log"
python3 "${WORKSPACE}/test_llava_cpu.py" 2>&1 | tee "${LOG_FILE}"

TEST_EXIT=$?

echo ""
echo "========================================================================="
if [ $TEST_EXIT -eq 0 ]; then
    echo "✓ Test completed successfully!"
else
    echo "✗ Test failed with exit code: $TEST_EXIT"
fi
echo "========================================================================="
echo "Log saved to: ${LOG_FILE}"
echo ""

exit $TEST_EXIT
