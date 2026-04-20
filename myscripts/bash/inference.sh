#!/usr/bin/env bash

set -eu -o pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

# Python は明示指定がなければ LIBERO 用venvを優先
if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif [[ -x "$ROOT_DIR/examples/libero/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/examples/libero/.venv/bin/python"
elif [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
else
  PYTHON_BIN="$(command -v python)"
fi

HOST="0.0.0.0"
PORT="8000"

# 以前の TASKS_CSV は「カンマ区切りで複数タスクを渡すための文字列」です。
# 今は bash 配列で指定します。例: TASKS=(libero_goal libero_spatial)
TASKS=(libero_10 libero_90)
CONTROL_FREQS_CSV="20"

RESIZE_SIZE="224"
REPLAN_STEPS="5"
NUM_STEPS_WAIT="10"
NUM_TRIALS_PER_TASK="50"
SEED="7"

BASE_VIDEO_OUT_PATH="videos/initial"
BASE_CSV_OUT_PATH="csv/libero/initial"

RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
LOG_DIR="logs/libero_runs/$RUN_ID"

IFS=',' read -r -a CONTROL_FREQS <<< "$CONTROL_FREQS_CSV"

mkdir -p "$LOG_DIR"

echo "[INFO] start sequential evaluation"
echo "[INFO] python: $PYTHON_BIN"
echo "[INFO] tasks: ${TASKS[*]}"
echo "[INFO] control freqs: ${CONTROL_FREQS[*]}"
echo "[INFO] logs: $LOG_DIR"

for task in "${TASKS[@]}"; do
  for control_freq in "${CONTROL_FREQS[@]}"; do

    run_name="${task}_cf${control_freq}hz"
    log_file="$LOG_DIR/${run_name}.log"

    # 出力は /out_path/<task>/<RUN_ID>/ の形に統一
    video_path="${BASE_VIDEO_OUT_PATH}/${task}/${RUN_ID}"
    csv_path="${BASE_CSV_OUT_PATH}/${task}/${RUN_ID}"

    mkdir -p "$video_path" "$csv_path"

    echo "[RUN] $run_name"

    if ! "$PYTHON_BIN" examples/libero/main.py \
      --args.host "$HOST" \
      --args.port "$PORT" \
      --args.resize-size "$RESIZE_SIZE" \
      --args.replan-steps "$REPLAN_STEPS" \
      --args.control-freq "$control_freq" \
      --args.task-suite-name "$task" \
      --args.num-steps-wait "$NUM_STEPS_WAIT" \
      --args.num-trials-per-task "$NUM_TRIALS_PER_TASK" \
      --args.video-out-path "$video_path" \
      --args.csv-out-path "$csv_path" \
      --args.seed "$SEED" \
      >"$log_file" 2>&1; then
      echo "[ERROR] $run_name failed. See: $log_file"
      tail -n 80 "$log_file" || true
      exit 1
    fi

    echo "[DONE] $run_name"

  done
done

echo "[RESULT] all jobs finished"