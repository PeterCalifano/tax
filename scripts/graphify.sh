#!/usr/bin/env bash
# scripts/graphify.sh — rebuild the graphify knowledge graph for this repo.
#
# Run this after significant code or documentation changes to keep .graphify/
# in sync with the codebase before committing.
#
# Usage:
#   ./scripts/graphify.sh            # full rebuild from current directory
#   ./scripts/graphify.sh --update   # incremental: re-extract only new/changed files
#   ./scripts/graphify.sh --no-viz   # skip HTML visualization

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Ensure graphify is installed
if ! command -v graphify >/dev/null 2>&1; then
    echo "graphify not found — installing @sentropic/graphify..."
    npm install -g @sentropic/graphify
fi

echo "Running graphify on $REPO_ROOT ..."
graphify "$REPO_ROOT" "$@"

echo ""
echo "Knowledge graph updated in .graphify/"
echo "  .graphify/graph.json      — machine-readable graph (GraphRAG-ready)"
echo "  .graphify/GRAPH_REPORT.md — plain-language architecture summary"
echo "  .graphify/graph.html      — interactive visualization (open in browser)"
