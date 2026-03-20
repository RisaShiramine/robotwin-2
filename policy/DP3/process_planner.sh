#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash policy/DP3/process_planner.sh --zarr=/path/to/name.zarr [options]

Required:
  --zarr=PATH                 Path to the target DP3 zarr dataset.

Optional:
  --instruction-dir=PATH      Directory containing episode*.json instructions.
  --output-dir=PATH           Output directory for planner artifacts.
  --instruction-source=MODE   seen|unseen|seen_first|unseen_first. Default: seen_first
  --decompose-scope=SCOPE     episode_map|all. Default: episode_map
  --boundary-mode=MODE        zarr_velocity|uniform. Default: zarr_velocity
  --api-key-env=NAME          DeepSeek API key env var. Default: DEEPSEEK_API_KEY
  --model=NAME                DeepSeek model. Default: deepseek-chat
  --timeout=SECONDS           API timeout. Default: 180
  --max-retries=N             API retry count. Default: 8
  --max-completion-tokens=N   Max completion tokens. Default: 1200
  --retry-backoff-max=N       Max retry backoff seconds. Default: 16
  --disable-stage-normalization
  --skip-verify
  -h, --help

This script automates the planner-data pipeline:
  1. aggregate episode instructions
  2. run DeepSeek decomposition (resume enabled)
  3. build planner_labels.jsonl against the provided zarr
  4. verify planner/zarr alignment
EOF
}

die() {
  echo "[process_planner] ERROR: $*" >&2
  exit 1
}

log() {
  echo "[process_planner] $*"
}

ZARR_PATH=""
INSTRUCTION_DIR=""
OUTPUT_DIR=""
INSTRUCTION_SOURCE="seen_first"
DECOMPOSE_SCOPE="episode_map"
BOUNDARY_MODE="zarr_velocity"
API_KEY_ENV="DEEPSEEK_API_KEY"
MODEL_NAME="deepseek-chat"
TIMEOUT_SECONDS="180"
MAX_RETRIES="8"
MAX_COMPLETION_TOKENS="1200"
RETRY_BACKOFF_MAX="16"
DISABLE_STAGE_NORMALIZATION="0"
SKIP_VERIFY="0"

for arg in "$@"; do
  case "$arg" in
    --zarr=*) ZARR_PATH="${arg#*=}" ;;
    --instruction-dir=*) INSTRUCTION_DIR="${arg#*=}" ;;
    --output-dir=*) OUTPUT_DIR="${arg#*=}" ;;
    --instruction-source=*) INSTRUCTION_SOURCE="${arg#*=}" ;;
    --decompose-scope=*) DECOMPOSE_SCOPE="${arg#*=}" ;;
    --boundary-mode=*) BOUNDARY_MODE="${arg#*=}" ;;
    --api-key-env=*) API_KEY_ENV="${arg#*=}" ;;
    --model=*) MODEL_NAME="${arg#*=}" ;;
    --timeout=*) TIMEOUT_SECONDS="${arg#*=}" ;;
    --max-retries=*) MAX_RETRIES="${arg#*=}" ;;
    --max-completion-tokens=*) MAX_COMPLETION_TOKENS="${arg#*=}" ;;
    --retry-backoff-max=*) RETRY_BACKOFF_MAX="${arg#*=}" ;;
    --disable-stage-normalization) DISABLE_STAGE_NORMALIZATION="1" ;;
    --skip-verify) SKIP_VERIFY="1" ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown argument: ${arg}" ;;
  esac
done

[ -n "${ZARR_PATH}" ] || { usage; die "--zarr is required."; }
[ -d "${ZARR_PATH}" ] || die "zarr path does not exist: ${ZARR_PATH}"

ZARR_ABS="$(python - <<'PY' "${ZARR_PATH}"
from pathlib import Path
import sys
print(Path(sys.argv[1]).expanduser().resolve())
PY
)"

ZARR_STEM="$(basename "${ZARR_ABS}" .zarr)"
TASK_NAME="${ZARR_STEM%%-*}"

if [ -z "${OUTPUT_DIR}" ]; then
  OUTPUT_DIR="${REPO_ROOT}/policy/DP3/examples/${ZARR_STEM}_planner"
fi
mkdir -p "${OUTPUT_DIR}"

discover_instruction_dir() {
  local candidates=(
    "${OUTPUT_DIR}/instructions"
    "$(dirname "${ZARR_ABS}")/instructions"
    "$(dirname "${ZARR_ABS}")/../instructions"
    "${REPO_ROOT}/policy/DP3/examples/instructions"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -d "${candidate}" ] && compgen -G "${candidate}/episode*.json" > /dev/null; then
      echo "${candidate}"
      return 0
    fi
  done
  return 1
}

if [ -z "${INSTRUCTION_DIR}" ]; then
  if INSTRUCTION_DIR="$(discover_instruction_dir)"; then
    log "Auto-detected instruction dir: ${INSTRUCTION_DIR}"
  else
    die "Could not auto-detect instruction dir for ${TASK_NAME}. Please pass --instruction-dir=/path/to/episode_jsons."
  fi
fi

[ -d "${INSTRUCTION_DIR}" ] || die "instruction dir does not exist: ${INSTRUCTION_DIR}"
compgen -G "${INSTRUCTION_DIR}/episode*.json" > /dev/null || die "No episode*.json found under ${INSTRUCTION_DIR}"

AGGREGATED_JSON="${OUTPUT_DIR}/official_instruction_batch.json"
EPISODE_MAP_JSON="${OUTPUT_DIR}/official_episode_instruction_map.json"
DECOMPOSITION_INPUT_JSON="${OUTPUT_DIR}/official_decomposition_input.json"
DECOMPOSITION_JSON="${OUTPUT_DIR}/official_task_decomposition_output.json"
DECOMPOSITION_JSONL="${OUTPUT_DIR}/official_task_decomposition_output.jsonl"
PLANNER_LABELS_JSONL="${OUTPUT_DIR}/planner_labels.jsonl"
PLANNER_SUMMARY_JSON="${OUTPUT_DIR}/planner_labels_summary.json"
VERIFY_SUMMARY_JSON="${OUTPUT_DIR}/planner_labels_verify_summary.json"

log "zarr: ${ZARR_ABS}"
log "task_name: ${TASK_NAME}"
log "output_dir: ${OUTPUT_DIR}"

python "${REPO_ROOT}/policy/DP3/scripts/process_instruction_dir.py" \
  "${INSTRUCTION_DIR}" \
  --aggregated-output "${AGGREGATED_JSON}" \
  --episode-map-output "${EPISODE_MAP_JSON}" \
  --decomposition-input-output "${DECOMPOSITION_INPUT_JSON}" \
  --instruction-source "${INSTRUCTION_SOURCE}" \
  --decompose-scope "${DECOMPOSE_SCOPE}"

python "${REPO_ROOT}/policy/DP3/scripts/decompose_tasks.py" \
  "${DECOMPOSITION_INPUT_JSON}" \
  --output "${DECOMPOSITION_JSON}" \
  --jsonl-output "${DECOMPOSITION_JSONL}" \
  --api-key-env "${API_KEY_ENV}" \
  --model "${MODEL_NAME}" \
  --timeout "${TIMEOUT_SECONDS}" \
  --max-retries "${MAX_RETRIES}" \
  --max-completion-tokens "${MAX_COMPLETION_TOKENS}" \
  --retry-backoff-max "${RETRY_BACKOFF_MAX}" \
  --deduplicate \
  --resume

BUILD_ARGS=(
  python "${REPO_ROOT}/policy/DP3/scripts/build_planner_labels.py"
  --decomposition-file "${DECOMPOSITION_JSON}"
  --episode-instruction-map "${EPISODE_MAP_JSON}"
  --dp3-zarr "${ZARR_ABS}"
  --output "${PLANNER_LABELS_JSONL}"
  --summary-output "${PLANNER_SUMMARY_JSON}"
  --boundary-mode "${BOUNDARY_MODE}"
)

if [ "${DISABLE_STAGE_NORMALIZATION}" = "1" ]; then
  BUILD_ARGS+=(--disable-stage-normalization)
fi

"${BUILD_ARGS[@]}"

if [ "${SKIP_VERIFY}" != "1" ]; then
  python "${REPO_ROOT}/policy/DP3/scripts/verify_planner_labels.py" \
    --planner-labels "${PLANNER_LABELS_JSONL}" \
    --dp3-zarr "${ZARR_ABS}" \
    --episode-instruction-map "${EPISODE_MAP_JSON}" \
    --summary-output "${VERIFY_SUMMARY_JSON}"
fi

log "Done."
log "Decomposition JSON: ${DECOMPOSITION_JSON}"
log "Planner labels JSONL: ${PLANNER_LABELS_JSONL}"
log "Planner summary: ${PLANNER_SUMMARY_JSON}"
if [ "${SKIP_VERIFY}" != "1" ]; then
  log "Verify summary: ${VERIFY_SUMMARY_JSON}"
fi
