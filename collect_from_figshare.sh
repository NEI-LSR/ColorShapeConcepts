#!/usr/bin/env bash
set -euo pipefail

# Ensure required commands are present
for cmd in unzip rsync; do
  if ! command -v "$cmd" &> /dev/null; then
    echo "Error: '$cmd' not found. Please install it." >&2
    exit 1
  fi
done

DEST_DIR="data"
mkdir -p "$DEST_DIR"

shopt -s nullglob

count=0
for ZIP in *.zip; do
  count=$((count+1))
  echo "→ Extracting: $ZIP"
  TMPDIR=$(mktemp -d)

  # unzip into tmp
  unzip -qq "$ZIP" -d "$TMPDIR"

  # detect if there's exactly one top‐level directory
  entries=("$TMPDIR"/*)
  if [ "${#entries[@]}" -eq 1 ] && [ -d "${entries[0]}" ]; then
    SRC="${entries[0]}/"
  else
    SRC="$TMPDIR/"
  fi

  # merge contents, dropping that single root dir if present
  rsync -a --no-perms \
    --chmod=Du=rwx,Dg=rx,Do=rx,Fu=rw,Fg=r,Fo=r \
    "$SRC" "$DEST_DIR"/

  rm -rf "$TMPDIR"
done

if [ "$count" -eq 0 ]; then
  echo "No .zip files found in $(pwd)."
else
  echo "Merged $count archive(s) into '$DEST_DIR/'."
fi
