#!/usr/bin/env bash
set -euo pipefail

PRECISE_MATH=0
SRC=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --precise-math)
      PRECISE_MATH=1
      shift
      ;;
    -h|--help)
      echo "Usage:"
      echo "  $0 [--precise-math] <source_file.cu>"
      exit 0
      ;;
    -* )
      echo "Error: unknown option: $1"
      echo "Usage:"
      echo "  $0 [--precise-math] <source_file.cu>"
      exit 1
      ;;
    *)
      if [[ -n "${SRC}" ]]; then
        echo "Error: only one source file may be provided"
        echo "Usage:"
        echo "  $0 [--precise-math] <source_file.cu>"
        exit 1
      fi
      SRC="$1"
      shift
      ;;
  esac
done

if [[ -z "${SRC}" ]]; then
  echo "Usage:"
  echo "  $0 [--precise-math] <source_file.cu>"
  exit 1
fi

# Ensure the file ends with .cu
if [[ "${SRC}" != *.cu ]]; then
  echo "Error: source file must have a .cu extension"
  exit 1
fi

# Strip the .cu suffix to get the output binary name
OUT_NAME="${SRC%.cu}"

MATH_DEFINE=(-DX17_PRECISE_MATH=${PRECISE_MATH})
FAST_MATH_FLAGS=(--use_fast_math)
if [[ "${PRECISE_MATH}" == "1" ]]; then
  FAST_MATH_FLAGS=()
fi

python block_cpp_config.py

/usr/local/cuda-12.6/bin/nvcc \
  -arch=sm_86 \
  -std=c++20 \
  -Xptxas=-v \
  --ftz=true \
  --prec-div=true \
  --fmad=true \
  "${FAST_MATH_FLAGS[@]}" \
  -I /home/spock/prog/cutlass/tools/util/include/ \
  -I /home/spock/prog/cutlass/include/ \
  "${MATH_DEFINE[@]}" \
  --expt-relaxed-constexpr \
  -maxrregcount=255 \
  -ptx \
  -O3 \
  "${SRC}" \
  -lineinfo \
  -o "tmp/${OUT_NAME}".ptx \
  2> /dev/null || true

set -x
/usr/local/cuda-12.6/bin/nvcc \
  -arch=sm_86 \
  -std=c++20 \
  -Xptxas=-v \
  --ftz=true \
  --prec-div=true \
  --fmad=true \
  "${FAST_MATH_FLAGS[@]}" \
  -I /home/spock/prog/cutlass/tools/util/include/ \
  -I /home/spock/prog/cutlass/include/ \
  "${MATH_DEFINE[@]}" \
  --expt-relaxed-constexpr \
  -maxrregcount=255 \
  -O3 \
  "${SRC}" \
  -lineinfo \
  -o "tmp/${OUT_NAME}"

cuobjdump --dump-sass tmp/attn > tmp/attn.sass
