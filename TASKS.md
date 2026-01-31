# TASKS — Execute in order

Phase 0 — bootstrap
0.1 Create package skeleton + Makefile + pyproject + ruff + pytest (make test + make lint pass)

Phase 1 — metrics + baselines
1.1 Implement eval/metrics.py + unit tests
1.2 Implement baselines/qiskit_baselines.py + smoke tests

Phase 2 — QASMBench ingestion + harness
2.1 Loader expects local path; optional opt-in fetch target
2.2 make eval-small -> artifacts/results.csv

Phase 3 — routing environment
3.1 RoutingEnv state/mask/step -> valid routed circuits

Phase 4 — imitation learning
4.1 Record SabreSwap action traces
4.2 Train IL policy

Phase 5 — RL fine-tuning
5.1 Multi-objective reward (swaps + depth + noise proxy)
5.2 RL train from IL

Phase 6 — Qiskit integration
6.1 TransformationPass wrapper

Phase 7 — full eval + figures
7.1 make eval-full
7.2 plots + summary
