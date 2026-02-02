VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip
SKIP_RL?=0
HARDWARE_DRAWS?=10
WEIGHTED_RESULTS_NAME?=$(if $(filter $(HARDWARE_DRAWS),10),results_noise_unguarded_weighted.csv,results_noise_unguarded_weighted_hd.csv)
WEIGHTED_SUMMARY_NAME?=$(if $(filter $(HARDWARE_DRAWS),10),summary_noise_unguarded_weighted.csv,summary_noise_unguarded_weighted_hd.csv)
WEIGHTED_SUMMARY_STD_NAME?=$(if $(filter $(HARDWARE_DRAWS),10),summary_noise_unguarded_weighted_std.csv,summary_noise_unguarded_weighted_hd_std.csv)

.PHONY: setup test lint format eval-small eval-pressure eval-full eval-noise eval-noise-unguarded eval-noise-unguarded-residual eval-noise-unguarded-weighted eval-teacher reproduce-paper

setup:
	python3 -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -e "[dev]" || true
	$(PIP) install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:
	$(PY) -m ruff format .
	$(PY) -m ruff check --fix .

eval-small:
	$(PY) -m quantum_routing_rl.eval.run_eval --suite dev --out artifacts --results-name results_dev.csv --summary-name summary_dev.csv --seeds 13 --include-teacher --il-checkpoint artifacts/checkpoints/il_soft.pt --il-name il_soft --il-teacher-mix 0 --rl-checkpoint artifacts/checkpoints/rl_ppo.pt --rl-name rl_ppo
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/results_dev.csv --out artifacts/plots_dev

eval-pressure:
	$(PY) -m quantum_routing_rl.eval.run_eval --suite pressure --out artifacts --results-name results_pressure.csv --summary-name summary_pressure.csv --seeds 13 --include-teacher --il-checkpoint artifacts/checkpoints/il_soft.pt --il-name il_soft --il-teacher-mix 0 --rl-checkpoint artifacts/checkpoints/rl_ppo.pt --rl-name rl_ppo
	$(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_pressure.csv --baseline il_soft --max-ratio 1.25 --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline
	if [ -f artifacts/checkpoints/rl_ppo.pt ]; then $(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_pressure.csv --baseline rl_ppo --teacher teacher_sabre_like --max-ratio 1.5 --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline; fi
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/results_pressure.csv --out artifacts/plots_pressure

eval-teacher:
	$(PY) -m quantum_routing_rl.eval.run_teacher_eval --suite pressure --out artifacts --results-name results_teacher.csv --summary-name summary_teacher.csv --seeds 13 --pressure-seed 99 --pressure-qasm 6 --hardware-samples 1
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/results_teacher.csv --out artifacts/plots_teacher

eval-full:
	$(PY) -m quantum_routing_rl.eval.run_eval --suite full --out artifacts --seeds 13 17 23 --results-name results.csv --summary-name summary.csv --include-teacher --il-checkpoint artifacts/checkpoints/il_soft.pt --il-name il_soft --il-teacher-mix 0 --rl-checkpoint artifacts/checkpoints/rl_ppo.pt --rl-name rl_ppo
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/results.csv --out artifacts/plots

eval-noise:
	$(PY) -m quantum_routing_rl.eval.run_eval --suite pressure --out artifacts --results-name results_noise.csv --summary-name summary_noise.csv --seeds 13 --hardware-samples 10 --hardware-seed-base 211 --include-teacher --hardware-profile realistic --hardware-snapshots 2 --hardware-drift 0.05 --hardware-directional --hardware-snapshot-spacing 50000 --hardware-crosstalk 0.01 --il-checkpoint artifacts/checkpoints/il_soft.pt --il-name il_soft --il-teacher-mix 0 --rl-checkpoint artifacts/checkpoints/rl_ppo.pt --rl-name rl_ppo
	$(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_noise.csv --baseline il_soft --max-ratio 1.25 --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline
	if [ -f artifacts/checkpoints/rl_ppo.pt ]; then $(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_noise.csv --baseline rl_ppo --teacher teacher_sabre_like --max-ratio 1.5 --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline; fi
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/results_noise.csv --out artifacts/plots_noise

eval-noise-unguarded:
	RL_ARGS=""; if [ "$(SKIP_RL)" != "1" ]; then RL_ARGS="--rl-checkpoint artifacts/checkpoints/rl_ppo.pt --rl-name rl_ppo --rl-no-fallback"; fi; \
	$(PY) -m quantum_routing_rl.eval.run_eval --suite pressure --out artifacts --results-name results_noise_unguarded.csv --summary-name summary_noise_unguarded.csv --summary-std-name summary_noise_unguarded_std.csv --seeds 13 17 23 --hardware-samples 10 --hardware-seed-base 211 --include-teacher --hardware-profile realistic --hardware-snapshots 2 --hardware-drift 0.05 --hardware-directional --hardware-snapshot-spacing 50000 --hardware-crosstalk 0.01 --il-checkpoint artifacts/checkpoints/il_soft.pt --il-name il_soft --il-teacher-mix 0 --il-no-fallback $$RL_ARGS
	$(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_noise_unguarded.csv --baseline il_soft --max-ratio 1.3 --metrics swaps_inserted twoq_depth total_duration_ns --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline
	if [ "$(SKIP_RL)" != "1" ] && [ -f artifacts/checkpoints/rl_ppo.pt ]; then $(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_noise_unguarded.csv --baseline rl_ppo --teacher teacher_sabre_like --max-ratio 1.3 --metrics swaps_inserted twoq_depth total_duration_ns --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline; fi
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/results_noise_unguarded.csv --out artifacts/plots_noise_unguarded

eval-noise-unguarded-residual:
	$(PY) -m quantum_routing_rl.eval.run_eval --suite pressure --out artifacts --results-name results_noise_unguarded.csv --summary-name summary_noise_unguarded.csv --summary-std-name summary_noise_unguarded_std.csv --seeds 13 17 23 --hardware-samples 10 --hardware-seed-base 211 --include-teacher --hardware-profile realistic --hardware-snapshots 2 --hardware-drift 0.05 --hardware-directional --hardware-snapshot-spacing 50000 --hardware-crosstalk 0.01 --il-checkpoint artifacts/checkpoints/il_soft.pt --il-name il_soft --il-teacher-mix 0 --il-no-fallback --residual-checkpoint artifacts/checkpoints/residual_topk.pt --residual-top-k 8 --residual-no-fallback
	$(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_noise_unguarded.csv --baseline il_soft --max-ratio 1.3 --metrics swaps_inserted twoq_depth total_duration_ns --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/results_noise_unguarded.csv --out artifacts/plots_noise_unguarded

eval-noise-unguarded-weighted:
	RL_ARGS=""; if [ "$(SKIP_RL)" != "1" ]; then RL_ARGS="--rl-checkpoint artifacts/checkpoints/rl_ppo.pt --rl-name rl_ppo --rl-no-fallback"; fi; \
	$(PY) -m quantum_routing_rl.eval.run_eval --suite pressure --out artifacts --results-name $(WEIGHTED_RESULTS_NAME) --summary-name $(WEIGHTED_SUMMARY_NAME) --summary-std-name $(WEIGHTED_SUMMARY_STD_NAME) --seeds 13 17 23 --hardware-samples $(HARDWARE_DRAWS) --hardware-seed-base 211 --include-teacher --hardware-profile realistic --hardware-snapshots 2 --hardware-drift 0.05 --hardware-directional --hardware-snapshot-spacing 50000 --hardware-crosstalk 0.01 --run-weighted-sabre --weighted-alpha-time 0.5 --weighted-beta-xtalk 0.2 --weighted-snapshot-mode avg --weighted-trials 8 --il-checkpoint artifacts/checkpoints/il_soft.pt --il-name il_soft --il-teacher-mix 0 --il-no-fallback $$RL_ARGS
	$(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/$(WEIGHTED_SUMMARY_NAME) --baseline il_soft --max-ratio 1.3 --metrics swaps_inserted twoq_depth total_duration_ns --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline
	$(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/$(WEIGHTED_SUMMARY_NAME) --baseline weighted_sabre --teacher qiskit_sabre_best --max-ratio 1.3 --metrics swaps_inserted twoq_depth total_duration_ns --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline
	if [ "$(SKIP_RL)" != "1" ] && [ -f artifacts/checkpoints/rl_ppo.pt ]; then $(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/$(WEIGHTED_SUMMARY_NAME) --baseline rl_ppo --teacher teacher_sabre_like --max-ratio 1.3 --metrics swaps_inserted twoq_depth total_duration_ns --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline; fi
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/$(WEIGHTED_RESULTS_NAME) --out artifacts/plots_noise_unguarded_weighted

reproduce-paper:
	@echo "Running weighted SABRE pressure eval (HARDWARE_DRAWS=50, SKIP_RL=1)..."
	SKIP_RL=1 HARDWARE_DRAWS=50 $(MAKE) eval-noise-unguarded-weighted
	@echo "Computing paired deltas, significance, and effect sizes..."
	$(PY) -m quantum_routing_rl.eval.paired_deltas --results artifacts/results_noise_unguarded_weighted_hd.csv --out artifacts/deltas --plots-out artifacts/plots_paired_deltas --stats-out artifacts/statistics
	@echo "Variance breakdown for Weighted vs Qiskit SABRE..."
	$(PY) -m quantum_routing_rl.eval.variance --results artifacts/results_noise_unguarded_weighted_hd.csv --out artifacts/variance --baselines weighted_sabre qiskit_sabre_best
	@echo "Ablation sweep (A0-A3) – skips rerun if results exist..."
	$(PY) -m quantum_routing_rl.eval.ablation --out artifacts --hardware-draws 30 --snapshot-mode avg --trials 8
	@echo "Yield vs overhead plot points..."
	$(PY) -m quantum_routing_rl.eval.yield_overhead --summary artifacts/summary_noise_unguarded_weighted_hd.csv --out artifacts/plots_yield_overhead --weighted-name weighted_sabre --baseline-name qiskit_sabre_best
	@echo "Paper tables + final figures..."
	$(PY) -m quantum_routing_rl.eval.paper_assets --results artifacts/results_noise_unguarded_weighted_hd.csv --summary artifacts/summary_noise_unguarded_weighted_hd.csv --variance artifacts/variance/variance_breakdown.csv
