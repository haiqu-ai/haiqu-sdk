"""
Haiqu SDK Optimization Module

Public problem type is ``qiskit_addon_opt_mapper.problems.OptimizationProblem``.
All ``OptimizationProblem`` inputs use the ``polynomial_problem`` payload. ``QUBO`` remains as
a deprecated LP compatibility surface.
"""

from qiskit_addon_opt_mapper.problems import OptimizationProblem

from .problem import (
    deserialize_optimization_problem,
    evaluate_problem_cost,
    from_hamiltonian,
    has_higher_order,
    is_constrained,
    max_pauli_weight,
    serialize_optimization_problem,
    to_minimize_problem,
    to_unconstrained_problem,
    uses_polynomial_wire,
    validate_optimization_problem,
)
from .qubo import QUBO
from .result import SolverResult
from .postprocess import cvar_expectation

__all__ = [
    "OptimizationProblem",
    "deserialize_optimization_problem",
    "evaluate_problem_cost",
    "from_hamiltonian",
    "has_higher_order",
    "is_constrained",
    "max_pauli_weight",
    "serialize_optimization_problem",
    "to_minimize_problem",
    "to_unconstrained_problem",
    "uses_polynomial_wire",
    "validate_optimization_problem",
    "QUBO",
    "SolverResult",
    "cvar_expectation",
]
