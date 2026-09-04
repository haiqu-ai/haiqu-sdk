"""Post-processing utilities for quantum optimization results.

This module provides utility functions for educational and analysis purposes.
For production postprocessing, use haiqu.postprocess() which leverages the Haiqu API.
"""

from __future__ import annotations
from typing import Dict, Callable, Union, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .qubo import QUBO
    from qiskit_addon_opt_mapper.problems import OptimizationProblem


def cvar_expectation(
    counts: Dict[str, Union[int, float]],
    problem: Union["OptimizationProblem", "QUBO", None] = None,
    alpha: float = 1.0,
    cost_function: Optional[Callable[[str], float]] = None,
) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) expectation value.

    This is a utility function for analyzing optimization results.

    Args:
        counts: Dictionary mapping bitstrings to counts/probabilities.
                Bitstrings in Qiskit convention (little-endian).
        problem: Unconstrained binary ``OptimizationProblem`` or deprecated ``QUBO``.
                 Required if cost_function is not provided.
        alpha: CVaR parameter (``0 < alpha <= 1``). Defaults to 1.0 (full expectation).
        cost_function: Custom cost function. If None, uses problem.cost().
                      Required if problem is not provided.

    Returns:
        CVaR expectation value.

    Raises:
        ValueError: If both problem and cost_function are None.

    Note:
        Bitstrings use Qiskit convention: rightmost bit = qubit 0.

    Examples:
        Using an OptimizationProblem:

        >>> cvar = cvar_expectation(counts, problem=problem, alpha=0.1)

        Using a custom cost function:

        >>> def my_cost(bitstring):
        ...     return sum(int(b) for b in bitstring)
        >>> cvar = cvar_expectation(counts, cost_function=my_cost, alpha=0.1)
    """
    # Validate that at least one method to compute cost is provided
    if cost_function is None and problem is None:
        raise ValueError(
            "Either 'problem' or 'cost_function' must be provided to evaluate costs.\n\n"
            "Examples:\n"
            "  # Using an OptimizationProblem:\n"
            "  cvar = cvar_expectation(counts, problem=problem)\n\n"
            "  # Using a custom cost function:\n"
            "  def cost_fn(bitstring):\n"
            "      return sum(int(b) for b in bitstring)\n"
            "  cvar = cvar_expectation(counts, cost_function=cost_fn)"
        )

    # Set up cost function
    if cost_function is None:
        from .problem import _objective_sense, evaluate_problem_cost

        sense = _objective_sense(problem)

        def cost_func(bs: str) -> float:
            return evaluate_problem_cost(problem, bs)

    else:
        sense = "min"
        cost_func = cost_function

    if alpha == 1.0:
        total = 0.0
        for bitstring, prob in counts.items():
            total += cost_func(bitstring) * prob
        return total

    nshots = sum(counts.values())
    cost_distribution = []
    for bitstring, prob in counts.items():
        cost = cost_func(bitstring)
        cost_distribution.append([cost, prob])

    reverse = sense == "max"
    sorted_costs = sorted(cost_distribution, key=lambda x: x[0], reverse=reverse)
    cvar = 0.0
    total_prob = 0.0

    for cost, prob in sorted_costs:
        cvar += cost * prob
        total_prob += prob
        if total_prob >= alpha * nshots:
            return cvar / total_prob

    return cvar / total_prob if total_prob > 0 else 0.0


__all__ = [
    "cvar_expectation",
]
