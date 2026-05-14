#!/bin/bash
# Create a timestamped summary file every INTERVAL seconds (default 3600 = 1 hour)
INTERVAL=${1:-3600}
OUT_DIR=logs/hourly
mkdir -p "$OUT_DIR"

while true; do
  LOG=$(ls -1t logs/docker_eval_*.log 2>/dev/null | head -n1)
  TS=$(date +"%Y%m%d_%H%M%S")
  OUT_FILE="$OUT_DIR/hourly_summary_${TS}.log"
  echo "[${TS}] Generating summary from: ${LOG}" > "$OUT_FILE"
  if [ -z "$LOG" ]; then
    echo "No docker_eval log found" >> "$OUT_FILE"
  else
    CHUNK=$(tail -n 2000 "$LOG" 2>/dev/null || true)
    echo "Recent lines from log (tail 2000):" >> "$OUT_FILE"
    echo "$CHUNK" >> "$OUT_FILE"
    echo >> "$OUT_FILE"
    echo "Counts (last 2000 lines):" >> "$OUT_FILE"
    echo "  TEST_STEP_START: $(echo "$CHUNK" | grep -c "\[TEST_STEP_START\")" >> "$OUT_FILE"
    echo "  TEST_STEP_BATCH_LOADED: $(echo "$CHUNK" | grep -c "\[TEST_STEP_BATCH_LOADED\")" >> "$OUT_FILE"
    echo "  TEST_STEP_INPUT_PREPARED: $(echo "$CHUNK" | grep -c "\[TEST_STEP_INPUT_PREPARED\")" >> "$OUT_FILE"
    echo "  TEST_STEP_INFER_DONE: $(echo "$CHUNK" | grep -c "\[TEST_STEP_INFER_DONE\")" >> "$OUT_FILE"
    echo "  TEST_STEP_COMPLETE: $(echo "$CHUNK" | grep -c "\[TEST_STEP_COMPLETE\")" >> "$OUT_FILE"
    echo >> "$OUT_FILE"
    echo "Recent progress lines:" >> "$OUT_FILE"
    tail -n 200 "$LOG" | grep -E "Testing:|\[TEST_STEP|infer_time|TEST_STEP_INFER_DONE|TEST_STEP_COMPLETE" | tail -n 100 >> "$OUT_FILE"
    echo >> "$OUT_FILE"
    echo "Recent errors (last 50 lines):" >> "$OUT_FILE"
    echo "$CHUNK" | grep -E "Traceback|Exception|Error" | tail -n 50 >> "$OUT_FILE" || true
  fi
  echo "Wrote $OUT_FILE"
  sleep $INTERVAL
done
