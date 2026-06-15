# MEMORY.md — Persistent AI Workflow Notes for `tax`

## Before Committing Significant Changes

Run graphify to keep the knowledge graph current:

```bash
./scripts/graphify.sh --update
```

Use `--update` for incremental re-extraction of only new/changed files (fast).
Use `./scripts/graphify.sh` (no flag) for a full rebuild after large refactors.

Commit the updated `.graphify/` artifacts (except the entries in `.gitignore`) alongside your code changes so the graph reflects the new state.

## When to Run a Full Rebuild

- After adding a new module or namespace (e.g., new `tax::*` submodule)
- After renaming or reorganizing files/directories
- After substantial changes to the ODE, ADS, or core kernel layers
- When `graphify query` returns stale or confusing results

## Querying the Knowledge Graph

```bash
graphify query "how does the Cauchy product dispatch work"
graphify path "TaylorExpansion" "AdsTree"
graphify explain "NliCriterion"
graphify summary --graph .graphify/graph.json
```

These return scoped subgraphs — much faster than grepping raw files.

## What the Graph Covers

The `.graphify/graph.json` at the repo root encodes:
- All C++ header relationships extracted via AST (Tree-sitter)
- Semantic concepts from documentation, tutorials, and CMakeLists files
- Community clusters with labels (kernel layer, ODE module, ADS module, etc.)
- God nodes (highest-betweenness concepts bridging modules)

See `.graphify/GRAPH_REPORT.md` for the plain-language architecture summary.
