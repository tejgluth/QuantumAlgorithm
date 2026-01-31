Read AGENTS.md and TASKS.md.

Execute TASKS.md from Phase 0 through Phase 7, in order.

After each phase: run the relevant make targets, summarize changes, and record the commands you ran + outcomes.
Do not skip baselines or benchmarks.
If Qiskit APIs differ by version, adapt and document behavior in artifacts/metadata.json and README.
Never fabricate results.

Start with Phase 0.1: create the full Python package skeleton under src/quantum_routing_rl/ matching the architecture described in AGENTS.md, add placeholder modules and minimal tests so `make test` and `make lint` pass, then proceed to Phase 1.
