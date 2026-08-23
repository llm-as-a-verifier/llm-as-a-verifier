#!/usr/bin/env bash
# Development-mode dependency link (macOS/Linux). See link_dev_deps.ps1 for
# the full rationale: point node_modules at a pnpm-installed deepseek-harness
# checkout's virtual store so the plugin resolves the same package versions
# the checkout runtime uses.
#
# Usage:
#   scripts/link_dev_deps.sh /path/to/deepseek-harness
set -euo pipefail

REPO_ROOT="${1:-${DSH_REPO_ROOT:-}}"
if [[ -z "$REPO_ROOT" ]]; then
  echo "usage: $0 <deepseek-harness checkout>  (or set DSH_REPO_ROOT)" >&2
  exit 2
fi

REPO="$(cd "$REPO_ROOT" && pwd)"
STORE="$REPO/node_modules/.pnpm/node_modules"
if [[ ! -d "$STORE/@deepseek-ai" ]]; then
  echo "error: pnpm virtual store not found under $REPO - run 'pnpm install' in the checkout first." >&2
  exit 1
fi

PLUGIN="$(cd "$(dirname "$0")/.." && pwd)"
STORE_DIR="$PLUGIN/.dev-store"
LINK="$PLUGIN/node_modules"

if [[ -e "$LINK" || -L "$LINK" ]]; then
  if [[ -L "$LINK" ]]; then
    echo "dev link already present: $LINK"
    exit 0
  fi
  echo "error: $LINK exists and is not a symlink; remove it manually first." >&2
  exit 1
fi

mkdir -p "$STORE_DIR"
ln -s "$STORE/@deepseek-ai" "$STORE_DIR/@deepseek-ai"
if [[ -d "$REPO/node_modules/@types" ]]; then
  ln -s "$REPO/node_modules/@types" "$STORE_DIR/@types"
fi
if [[ -d "$REPO/node_modules/typescript" ]]; then
  ln -s "$REPO/node_modules/typescript" "$STORE_DIR/typescript"
fi
if [[ -d "$REPO/node_modules/.bin" ]]; then
  ln -s "$REPO/node_modules/.bin" "$STORE_DIR/.bin"
fi
ln -s "$STORE_DIR" "$LINK"
echo "linked $LINK -> $STORE_DIR"
