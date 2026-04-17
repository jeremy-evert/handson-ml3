#!/usr/bin/env bash
set -u

# Run Stage 1 prompt generation + Codex execution for chapter notebooks 01..19.
#
# Usage:
#   bash tools/notebook_enricher/run_stage1_batch.sh
#   bash tools/notebook_enricher/run_stage1_batch.sh --clean-tmp
#
# Behavior:
# - calls stage1_prompt_wrapper.py for each chapter notebook
# - if a prompt is generated, immediately runs tools/codex/run_prompt.py on it
# - skips notebooks the wrapper classifies as already substantive
# - continues on errors and prints a summary at the end

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR" || exit 1

WRAPPER="tools/notebook_enricher/stage1_prompt_wrapper.py"
RUNNER="tools/codex/run_prompt.py"

CLEAN_TMP=0
if [[ "${1:-}" == "--clean-tmp" ]]; then
  CLEAN_TMP=1
fi

if [[ ! -f "$WRAPPER" ]]; then
  echo "ERROR: wrapper not found: $WRAPPER" >&2
  exit 1
fi

if [[ ! -f "$RUNNER" ]]; then
  echo "ERROR: runner not found: $RUNNER" >&2
  exit 1
fi

if [[ "$CLEAN_TMP" -eq 1 ]]; then
  echo "Cleaning old .ipynb.tmp files..."
  find . -maxdepth 1 -type f -name "*.ipynb.tmp" -print -delete
  echo
fi

declare -a GENERATED_NOTEBOOKS=()
declare -a SKIPPED_NOTEBOOKS=()
declare -a FAILED_NOTEBOOKS=()

generated_count=0
skipped_count=0
failed_count=0

for i in $(seq -w 1 19); do
  notebook="${i}"_*".ipynb"

  # Expand glob safely
  matches=( $notebook )
  if [[ ${#matches[@]} -ne 1 || ! -f "${matches[0]}" ]]; then
    echo "[$i] ERROR: could not uniquely resolve notebook for pattern: $notebook"
    FAILED_NOTEBOOKS+=("$i:resolve")
    ((failed_count++))
    echo
    continue
  fi

  notebook="${matches[0]}"

  echo "=================================================================="
  echo "[$i] Processing $notebook"
  echo "=================================================================="

  wrapper_output="$(python3 "$WRAPPER" --overwrite-generated "$notebook" 2>&1)"
  wrapper_rc=$?

  echo "$wrapper_output"

  if [[ $wrapper_rc -ne 0 ]]; then
    echo "[$i] WRAPPER FAILED"
    FAILED_NOTEBOOKS+=("$notebook:wrapper")
    ((failed_count++))
    echo
    continue
  fi

  if grep -q '^SKIPPED ' <<< "$wrapper_output"; then
    echo "[$i] SKIPPED BY WRAPPER"
    SKIPPED_NOTEBOOKS+=("$notebook")
    ((skipped_count++))
    echo
    continue
  fi

  prompt_path="$(awk '/^GENERATED / {print $2; exit}' <<< "$wrapper_output")"

  if [[ -z "$prompt_path" ]]; then
    echo "[$i] ERROR: wrapper succeeded but no generated prompt path was found"
    FAILED_NOTEBOOKS+=("$notebook:no-generated-prompt")
    ((failed_count++))
    echo
    continue
  fi

  if [[ ! -f "$prompt_path" ]]; then
    echo "[$i] ERROR: generated prompt not found: $prompt_path"
    FAILED_NOTEBOOKS+=("$notebook:missing-prompt-file")
    ((failed_count++))
    echo
    continue
  fi

  echo
  echo "[$i] Running Codex on $prompt_path"
  runner_output="$(./"$RUNNER" "$prompt_path" 2>&1)"
  runner_rc=$?

  echo "$runner_output"

  if [[ $runner_rc -ne 0 ]]; then
    echo "[$i] RUNNER FAILED"
    FAILED_NOTEBOOKS+=("$notebook:runner")
    ((failed_count++))
    echo
    continue
  fi

  GENERATED_NOTEBOOKS+=("$notebook")
  ((generated_count++))
  echo "[$i] DONE"
  echo
done

echo "=================================================================="
echo "SUMMARY"
echo "=================================================================="
echo "Generated and ran: $generated_count"
for item in "${GENERATED_NOTEBOOKS[@]}"; do
  echo "  - $item"
done

echo
echo "Skipped: $skipped_count"
for item in "${SKIPPED_NOTEBOOKS[@]}"; do
  echo "  - $item"
done

echo
echo "Failed: $failed_count"
for item in "${FAILED_NOTEBOOKS[@]}"; do
  echo "  - $item"
done

echo
if [[ $failed_count -gt 0 ]]; then
  exit 1
fi
exit 0
