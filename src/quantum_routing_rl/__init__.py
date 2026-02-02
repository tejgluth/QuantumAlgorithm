"""quantum-routing-rl package."""

from quantum_routing_rl.baselines.qiskit_baselines import (
    BaselineResult,
    run_best_available_sabre,
    run_sabre_layout_swap,
)
from quantum_routing_rl.benchmarks.qasmbench_loader import QasmCircuit, load_suite
from quantum_routing_rl.hardware.model import EdgeProps, HardwareModel, QubitProps
from quantum_routing_rl.models.policy import SwapPolicy, load_swap_policy, route_with_policy
from quantum_routing_rl.models.teacher import TeacherPolicy, TeacherWeights, route_with_teacher
from quantum_routing_rl.models.weighted_sabre import (
    WeightedDistanceParams,
    WeightedSabreWeights,
    route_with_weighted_sabre,
)
from quantum_routing_rl.eval.run_eval import main as run_eval

__all__ = [
    "BaselineResult",
    "QasmCircuit",
    "EdgeProps",
    "HardwareModel",
    "QubitProps",
    "SwapPolicy",
    "TeacherPolicy",
    "TeacherWeights",
    "load_swap_policy",
    "route_with_teacher",
    "load_suite",
    "route_with_policy",
    "run_best_available_sabre",
    "run_sabre_layout_swap",
    "route_with_weighted_sabre",
    "WeightedDistanceParams",
    "WeightedSabreWeights",
    "run_eval",
    "make_synthetic_hardware",
]


def make_synthetic_hardware(graph, *, seed: int, profile: str = "realistic", **kwargs):
    """Helper to build a synthetic HardwareModel from a NetworkX graph."""
    return HardwareModel.synthetic(graph, seed=seed, profile=profile, **kwargs)
