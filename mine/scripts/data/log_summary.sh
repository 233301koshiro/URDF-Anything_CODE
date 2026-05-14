#!/bin/bash
# Periodically summarize the latest docker_eval log (default interval 300s)
INTERVAL=${1:-300}
SUMMARY_LOG=logs/eval_summary.log
mkdir -p logs

while true; do
  LOG=$(ls -1t logs/docker_eval_*.log 2>/dev/null | head -n1)
  TS=$(date +"%Y-%m-%d %H:%M:%S")
  if [ -z "$LOG" ]; then
    echo "[$TS] No docker_eval log found" >> "$SUMMARY_LOG"
    sleep $INTERVAL
    continue
  fi

  CHUNK=$(tail -n 1000 "$LOG" 2>/dev/null || true)
  TEST_STARTS=$(echo "$CHUNK" | grep -c "\[TEST_STEP_START\")
  BATCH_LOADED=$(echo "$CHUNK" | grep -c "\[TEST_STEP_BATCH_LOADED\")
  INPUT_PREP=$(echo "$CHUNK" | grep -c "\[TEST_STEP_INPUT_PREPARED\")
  INFER_DONE=$(echo "$CHUNK" | grep -c "\[TEST_STEP_INFER_DONE\")
  COMPLETE=$(echo "$CHUNK" | grep -c "\[TEST_STEP_COMPLETE\")
  ERRORS=$(echo "$CHUNK" | grep -E "Traceback|Exception|Error" | tail -n 20)
  PROGRESS=$(tail -n 200 "$LOG" | grep -E "Testing:|\[TEST_STEP|infer_time|TEST_STEP_INFER_DONE|TEST_STEP_COMPLETE" | tail -n 10)

  echo "[$TS] Log: $LOG" >> "$SUMMARY_LOG"
  echo "  Recent counts (last 1000 lines):" >> "$SUMMARY_LOG"
  echo "    TEST_STEP_START: $TEST_STARTS" >> "$SUMMARY_LOG"
  echo "    TEST_STEP_BATCH_LOADED: $BATCH_LOADED" >> "$SUMMARY_LOG"
  echo "    TEST_STEP_INPUT_PREPARED: $INPUT_PREP" >> "$SUMMARY_LOG"
  echo "    TEST_STEP_INFER_DONE: $INFER_DONE" >> "$SUMMARY_LOG"
  echo "    TEST_STEP_COMPLETE: $COMPLETE" >> "$SUMMARY_LOG"
  if [ -n "$ERRORS" ]; then
    echo "  Recent errors:" >> "$SUMMARY_LOG"
    echo "$ERRORS" >> "$SUMMARY_LOG"
  fi
  if [ -n "$PROGRESS" ]; then
    echo "  Recent progress lines:" >> "$SUMMARY_LOG"
    echo "$PROGRESS" >> "$SUMMARY_LOG"
  fi
  echo "----" >> "$SUMMARY_LOG"

  sleep $INTERVAL
done
