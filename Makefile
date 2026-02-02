VENV?=.venv
PY?=$(VENV)/bin/python
PIP?=$(VENV)/bin/pip
SKIP_RL?=0

.PHONY: setup test lint format eval-small eval-pressure eval-full eval-noise eval-noise-unguarded eval-noise-unguarded-residual eval-noise-unguarded-weighted eval-teacher

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
	$(PY) -m quantum_routing_rl.eval.run_eval --suite pressure --out artifacts --results-name results_noise_unguarded_weighted.csv --summary-name summary_noise_unguarded_weighted.csv --summary-std-name summary_noise_unguarded_weighted_std.csv --seeds 13 17 23 --hardware-samples 10 --hardware-seed-base 211 --include-teacher --hardware-profile realistic --hardware-snapshots 2 --hardware-drift 0.05 --hardware-directional --hardware-snapshot-spacing 50000 --hardware-crosstalk 0.01 --run-weighted-sabre --weighted-alpha-time 0.5 --weighted-beta-xtalk 0.2 --weighted-snapshot-mode avg --weighted-trials 8 --il-checkpoint artifacts/checkpoints/il_soft.pt --il-name il_soft --il-teacher-mix 0 --il-no-fallback $$RL_ARGS
	$(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_noise_unguarded_weighted.csv --baseline il_soft --max-ratio 1.3 --metrics swaps_inserted twoq_depth total_duration_ns --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline
	$(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_noise_unguarded_weighted.csv --baseline weighted_sabre --teacher qiskit_sabre_best --max-ratio 1.3 --metrics swaps_inserted twoq_depth total_duration_ns --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline
	if [ "$(SKIP_RL)" != "1" ] && [ -f artifacts/checkpoints/rl_ppo.pt ]; then $(PY) -m quantum_routing_rl.eval.regression_checks --summary artifacts/summary_noise_unguarded_weighted.csv --baseline rl_ppo --teacher teacher_sabre_like --max-ratio 1.3 --metrics swaps_inserted twoq_depth total_duration_ns --graphs ring_8 grid_3x3 heavy_hex_15 --require-baseline; fi
	$(PY) -m quantum_routing_rl.eval.plots --in artifacts/results_noise_unguarded_weighted.csv --out artifacts/plots_noise_unguarded_weighted
