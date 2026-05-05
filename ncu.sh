#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage:"
  echo "	$0 <path_to_executable>"
  exit 1
fi

TARGET="$1"
OUT_FILE="${TARGET}.ncu-rep"

set -x
/usr/local/cuda-12.6/bin/ncu \
	--target-processes all \
	--set full \
	--launch-skip 50 \
	--launch-count 1 \
	-o "${OUT_FILE}" \
	-f \
	"${TARGET}"

/usr/local/cuda-12.6/bin/ncu \
	--import "${OUT_FILE}" \
	--section SpeedOfLight \
	--section MemoryWorkloadAnalysis \
	--section MemoryWorkloadAnalysis_Tables \
	--section LaunchStats \
	--section Occupancy
