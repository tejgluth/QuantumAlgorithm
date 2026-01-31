# AGENTS.md — Project Instructions for Codex

## Mission
Build a publishable, reproducible research codebase that:
1) Benchmarks Qiskit SABRE-family routing baselines (SabreLayout + SabreSwap; plus best SABRE-family routing available in installed Qiskit).
2) Implements a learned routing policy (IL + RL) that improves at least one of: SWAPs, 2Q count/2Q depth, noise-aware success proxy.
3) Produces paper-quality figures and an evaluation report reproducible from a single command.

## Non-negotiable rules
- Never run destructive commands (rm -rf, wipe git history, delete outside repo).
- Never commit secrets. Keep `.env` ignored.
- Small, test-backed commits. Don’t fabricate results.

## Workflow contracts
Use Makefile targets: make setup, make test, make lint, make eval-small, make eval-full

## Implementation strategy (must follow order)
1) Harness + metrics + baselines
2) Deterministic baseline checks
3) Routing env -> valid routed circuits
4) Imitation learning from SABRE traces
5) RL fine-tuning (multi-objective/noise-aware)
6) Qiskit TransformationPass wrapper
7) Full eval + plots + summary
8) Paper-style writeup in paper/

## Completion definition
Done when `make eval-full` produces artifacts/results.csv, artifacts/summary.csv, and plots (png+pdf) for swaps, 2Q depth, noise proxy, runtime, Pareto.
