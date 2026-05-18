#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build a QuickUMLS index from an installed UMLS Metathesaurus directory.

Usage:
  scripts/build_quickumls_index.sh <umls_dir> [quickumls_out_dir]

Arguments:
  umls_dir            Directory containing MRCONSO.RRF and MRSTY.RRF.
  quickumls_out_dir   Output index directory. Default: data/quickumls/2024AB

Example from WSL:
  cd /mnt/c/Users/dsw54/Desktop/codex_related/flow_ehr
  . .venv-eval/bin/activate
  scripts/build_quickumls_index.sh data/umls/2024AB/META data/quickumls/2024AB

Then evaluate with:
  export QUICKUMLS_PATH="$PWD/data/quickumls/2024AB"
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || $# -lt 1 ]]; then
  usage
  exit 0
fi

UMLS_DIR="$1"
OUT_DIR="${2:-data/quickumls/2024AB}"

if [[ ! -f "$UMLS_DIR/MRCONSO.RRF" ]]; then
  echo "Missing $UMLS_DIR/MRCONSO.RRF" >&2
  exit 1
fi

if [[ ! -f "$UMLS_DIR/MRSTY.RRF" ]]; then
  echo "Missing $UMLS_DIR/MRSTY.RRF" >&2
  exit 1
fi

mkdir -p "$(dirname "$OUT_DIR")"
python -m quickumls.install "$UMLS_DIR" "$OUT_DIR" -d leveldb -E ENG

echo "QuickUMLS index created at: $OUT_DIR"
echo "Use: export QUICKUMLS_PATH=\"$(cd "$(dirname "$OUT_DIR")" && pwd)/$(basename "$OUT_DIR")\""
