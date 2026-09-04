"""Public combinatorial-optimization intake around ``OptimizationProblem``.

Haiqu accepts unconstrained binary ``OptimizationProblem`` instances of any
polynomial degree (QUBO or native HUBO). Supported ``OptimizationProblem``
inputs use the ``polynomial_problem`` payload. Only the deprecated Haiqu
``QUBO`` compatibility surface uses the legacy CPLEX LP wire.
"""

from __future__ import annotations

import warnings
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, Optional, Union

from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_opt_mapper.converters import OptimizationProblemToQubo
from qiskit_addon_opt_mapper.problems import OptimizationProblem

if TYPE_CHECKING:
    from .qubo import QUBO

_HUBO_REFUSED = (
    "This path only accepts an unconstrained quadratic OptimizationProblem "
    "(2-local diagonal Ising: I/Z/ZZ). Higher-order objectives must be "
    "reduced to QUBO first; that reduction adds extra variables. "
    "This function does not perform that reduction."
)

_CONSTRAINED_REFUSED = (
    "OptimizationProblem has constraints; "
    "call to_unconstrained_problem(op) first to fold penalties into an "
    "unconstrained binary problem, then pass that result here."
)


def max_pauli_weight(hamiltonian: SparsePauliOp) -> int:
    """Max number of non-I Paulis in any term (0 if empty)."""
    if not isinstance(hamiltonian, SparsePauliOp):
        raise TypeError(f"max_pauli_weight expects SparsePauliOp, got {type(hamiltonian)!r}")
    n = hamiltonian.num_qubits
    best = 0
    for pauli_str, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
        if abs(float(coeff.real)) < 1e-15 and abs(float(coeff.imag)) < 1e-15:
            continue
        weight = sum(1 for v in range(n) if pauli_str[n - 1 - v] != "I")
        best = max(best, weight)
    return best


def _assert_2local_diagonal_ising(hamiltonian: SparsePauliOp) -> None:
    if not isinstance(hamiltonian, SparsePauliOp):
        raise TypeError("Input must be a qiskit.quantum_info.SparsePauliOp.")
    n = hamiltonian.num_qubits
    best = 0
    for pauli_str, coeff in zip(hamiltonian.paulis.to_labels(), hamiltonian.coeffs):
        if abs(float(coeff.real)) < 1e-15 and abs(float(coeff.imag)) < 1e-15:
            continue
        if abs(float(coeff.imag)) > 1e-10:
            raise ValueError("Hamiltonian coefficients must be real.")
        weight = 0
        for v in range(n):
            op = pauli_str[n - 1 - v]
            if op in ("X", "Y"):
                raise ValueError("Hamiltonian must be diagonal Ising (I/Z only); " f"found Pauli '{op}' in term '{pauli_str}'.")
            if op != "I":
                weight += 1
        best = max(best, weight)
    if best > 2:
        raise ValueError(_HUBO_REFUSED)


def is_constrained(op: Any) -> bool:
    """True if linear, quadratic, or higher-order constraints are present on ``op``."""
    return bool(
        getattr(op, "linear_constraints", None)
        or getattr(op, "quadratic_constraints", None)
        or getattr(op, "higher_order_constraints", None)
    )


def has_higher_order(op: Any) -> bool:
    """True if ``op.objective.higher_order`` contains any terms."""
    objective = getattr(op, "objective", None)
    higher_order = getattr(objective, "higher_order", None) or {}
    return bool(higher_order)


def to_unconstrained_problem(op: OptimizationProblem, *, penalty: Optional[float] = None) -> OptimizationProblem:
    """Fold constraints into penalties and return an unconstrained ``OptimizationProblem``.

    Uses opt-mapper ``OptimizationProblemToQubo``. The returned objective is a
    penalized proxy of the original constrained problem, not the same objective.
    Feasibility checking stays classical / application-specific.

    ``penalty``:
      * ``None`` (default) — opt-mapper chooses its default penalty scale.
      * ``float`` — forwarded to ``OptimizationProblemToQubo(penalty=...)``.
    """
    if not isinstance(op, OptimizationProblem):
        raise TypeError(
            "to_unconstrained_problem expects qiskit_addon_opt_mapper.problems.OptimizationProblem, " f"got {type(op)!r}."
        )
    kwargs: dict[str, Any] = {}
    if penalty is not None:
        kwargs["penalty"] = float(penalty)
    return OptimizationProblemToQubo(**kwargs).convert(op)


def from_hamiltonian(H: SparsePauliOp, offset: float = 0.0) -> OptimizationProblem:
    """Create an unconstrained quadratic ``OptimizationProblem`` from a 2-local Ising Hamiltonian.

    Converts spin variables :math:`s_i \\in \\{-1,+1\\}` to binary :math:`x_i \\in \\{0,1\\}`
    via :math:`s_i = 1 - 2 x_i`. Higher-order Hamiltonians are refused: reduce
    the objective to QUBO first (that reduction adds extra variables). This
    helper reconstructs only quadratic objectives.

    Args:
        H: Diagonal Ising Hamiltonian (I/Z/ZZ terms only).
        offset: Constant added to the objective. Defaults to 0.0.

    Returns:
        Unconstrained quadratic ``OptimizationProblem`` ready for
        ``haiqu.build_lr_qaoa_circuit``, ``haiqu.run``, and ``haiqu.postprocess``.

    Raises:
        TypeError: If ``H`` is not a ``SparsePauliOp``.
        ValueError: If ``H`` contains X/Y, complex coefficients, or Pauli weight greater than 2.
    """
    _assert_2local_diagonal_ising(H)
    n = H.num_qubits
    linear = {i: 0.0 for i in range(n)}
    quadratic: dict[tuple[int, int], float] = {}
    constant = float(offset)

    for pauli_str, coeff in zip(H.paulis.to_labels(), H.coeffs):
        c = float(coeff.real)
        if abs(c) < 1e-15:
            continue
        qubits = [v for v in range(n) if pauli_str[n - 1 - v] != "I"]
        if len(qubits) == 0:
            constant += c
        elif len(qubits) == 1:
            i = qubits[0]
            constant += c
            linear[i] += -2.0 * c
        else:
            i, j = qubits[0], qubits[1]
            constant += c
            linear[i] += -2.0 * c
            linear[j] += -2.0 * c
            key = (min(i, j), max(i, j))
            quadratic[key] = quadratic.get(key, 0.0) + 4.0 * c

    op = OptimizationProblem("ising")
    if n == 0:
        raise ValueError("Hamiltonian must act on at least one qubit.")
    op.binary_var_list(n)
    op.minimize(constant=constant, linear=linear, quadratic=quadratic or None)
    return op


def validate_optimization_problem(op: OptimizationProblem) -> None:
    """Validate an unconstrained binary ``OptimizationProblem`` for Haiqu solvers.

    Accepts quadratic QUBO and native higher-order (HUBO) objectives. Optional
    helper: ``haiqu.build_lr_qaoa_circuit`` and ``haiqu.postprocess`` already
    run this validation, so callers can pass an unconstrained
    ``OptimizationProblem`` directly. Constrained applications must be
    penalty-folded with ``to_unconstrained_problem`` first.

    Args:
        op: Unconstrained binary ``OptimizationProblem`` of any polynomial degree.

    Raises:
        TypeError: If ``op`` is not an ``OptimizationProblem``.
        ValueError: If the problem has constraints or non-binary variables.
    """
    _validate_unconstrained_binary_problem(op)


def _validate_unconstrained_binary_problem(op: Any) -> None:
    """Allow any-degree unconstrained binary objectives (QUBO or HUBO)."""
    if not isinstance(op, OptimizationProblem):
        raise TypeError("Expected qiskit_addon_opt_mapper.problems.OptimizationProblem, " f"got {type(op)!r}.")
    if is_constrained(op):
        raise ValueError(_CONSTRAINED_REFUSED)
    n_vars = op.get_num_vars()
    n_binary = op.get_num_binary_vars()
    n_spin = op.get_num_spin_vars() if hasattr(op, "get_num_spin_vars") else 0
    if n_vars == 0:
        raise ValueError("OptimizationProblem must contain at least one variable.")
    if n_binary != n_vars or n_spin != 0:
        raise ValueError(
            "Haiqu solvers require binary variables only. "
            "Call to_unconstrained_problem(op) first if the problem has integer, "
            "continuous, or spin variables."
        )


def _validate_qubo_solver_problem(op: Any) -> None:
    """Quadratic-only gate for the LP / ``_as_qubo`` adapter path."""
    _validate_unconstrained_binary_problem(op)
    if has_higher_order(op):
        raise ValueError(_HUBO_REFUSED)
    hamiltonian, _offset = op.to_ising()
    if not isinstance(hamiltonian, SparsePauliOp):
        hamiltonian = SparsePauliOp(hamiltonian)
    if max_pauli_weight(hamiltonian) > 2:
        raise ValueError(_HUBO_REFUSED)


def _objective_sense(problem: Any) -> Literal["min", "max"]:
    """Return ``\"min\"`` or ``\"max\"`` from ``problem.objective.sense`` when present."""
    objective = getattr(problem, "objective", None)
    if objective is None or not hasattr(objective, "sense"):
        return "min"
    sense_name = getattr(objective.sense, "name", str(objective.sense)).upper()
    return "max" if "MAX" in sense_name else "min"


def to_minimize_problem(problem: Union[OptimizationProblem, "QUBO"]) -> Union[OptimizationProblem, "QUBO"]:
    """Return a MINIMIZE working copy. The original problem is not mutated.

    ``OptimizationProblem`` uses opt-mapper ``MaximizeToMinimize``.
    Deprecated ``QUBO`` uses the qiskit-optimization converter on ``problem._qp``,
    then wraps the result via the public ``QUBO.from_quadratic_program`` path.
    Problems already in MINIMIZE form are returned unchanged.
    """
    from qiskit_addon_opt_mapper.converters import MaximizeToMinimize

    from .qubo import QUBO

    if isinstance(problem, OptimizationProblem):
        if _objective_sense(problem) == "min":
            return problem
        return MaximizeToMinimize().convert(problem)
    if isinstance(problem, QUBO):
        if _objective_sense(problem._qp) == "min":
            return problem
        from qiskit_optimization.converters import MaximizeToMinimize as QpMaximizeToMinimize

        converted_qp = QpMaximizeToMinimize().convert(problem._qp)
        if converted_qp is problem._qp:
            return problem
        return QUBO.from_quadratic_program(converted_qp)
    raise TypeError(
        "problem must be qiskit_addon_opt_mapper.problems.OptimizationProblem "
        f"or haiqu.sdk.optimization.QUBO, got {type(problem)!r}."
    )


def _best_bitstring(costs: dict[str, float], sense: Literal["min", "max"]) -> tuple[str, float]:
    """Return the best bitstring and its cost for the given optimization sense."""
    if not costs:
        raise ValueError("costs must not be empty.")
    if sense == "min":
        best_key = min(costs, key=costs.get)
    else:
        best_key = max(costs, key=costs.get)
    return best_key, float(costs[best_key])


def _fold_binary_monomial(indices: tuple[int, ...]) -> tuple[int, ...]:
    """Sort binary indices and collapse duplicates (:math:`x_i^2 = x_i`)."""
    if not indices:
        return ()
    sorted_indices = sorted(int(i) for i in indices)
    folded: list[int] = []
    for idx in sorted_indices:
        if not folded or folded[-1] != idx:
            folded.append(idx)
    return tuple(folded)


def _normalize_binary_polynomial(op: OptimizationProblem, *, in_place: bool = False) -> OptimizationProblem:
    """Normalize diagonal quadratics and repeated higher-order monomials for binary variables."""
    if not isinstance(op, OptimizationProblem):
        raise TypeError(f"_normalize_binary_polynomial expects OptimizationProblem, got {type(op)!r}.")

    source = op if in_place else deepcopy(op)
    objective = source.objective
    constant = float(getattr(objective, "constant", 0.0))
    linear = {int(i): float(c) for i, c in objective.linear.to_dict().items()}
    quadratic = {(int(i), int(j)): float(c) for (i, j), c in objective.quadratic.to_dict().items()}

    for (i, j), coeff in list(quadratic.items()):
        if i == j:
            linear[i] = linear.get(i, 0.0) + coeff
            del quadratic[(i, j)]

    normalized_quadratic: dict[tuple[int, int], float] = {}
    for (i, j), coeff in quadratic.items():
        key = (min(i, j), max(i, j))
        normalized_quadratic[key] = normalized_quadratic.get(key, 0.0) + coeff

    folded_higher: dict[int, dict[tuple[int, ...], float]] = {}
    for _order, expr in (objective.higher_order or {}).items():
        mapping = expr.to_dict() if hasattr(expr, "to_dict") else dict(expr)
        for key, coeff in mapping.items():
            idxs = tuple(int(i) for i in key) if not isinstance(key, tuple) else tuple(int(i) for i in key)
            folded = _fold_binary_monomial(idxs)
            order_f = len(folded)
            c = float(coeff)
            if order_f == 0:
                constant += c
            elif order_f == 1:
                linear[folded[0]] = linear.get(folded[0], 0.0) + c
            elif order_f == 2:
                qkey = (min(folded[0], folded[1]), max(folded[0], folded[1]))
                normalized_quadratic[qkey] = normalized_quadratic.get(qkey, 0.0) + c
            else:
                bucket = folded_higher.setdefault(order_f, {})
                bucket[folded] = bucket.get(folded, 0.0) + c

    sense = _objective_sense(source)
    n_vars = source.get_num_vars()
    normalized = OptimizationProblem(getattr(source, "name", "normalized"))
    normalized.binary_var_list(n_vars)
    kwargs: dict[str, Any] = {
        "constant": constant,
        "linear": linear or None,
        "quadratic": normalized_quadratic or None,
    }
    if folded_higher:
        kwargs["higher_order"] = folded_higher
    if sense == "max":
        normalized.maximize(**kwargs)
    else:
        normalized.minimize(**kwargs)
    return normalized


def _index_key(indices: tuple[int, ...]) -> str:
    return ",".join(str(int(i)) for i in indices)


def _parse_index_key(key: str) -> tuple[int, ...]:
    parts = str(key).split(",")
    if not parts or any(part.strip() == "" for part in parts):
        raise ValueError(f"Invalid polynomial index key {key!r}.")
    return tuple(int(part) for part in parts)


def serialize_optimization_problem(op: OptimizationProblem) -> dict[str, Any]:
    """Serialize an unconstrained binary ``OptimizationProblem`` to a ``polynomial_problem`` payload.

    Args:
        op: Unconstrained binary ``OptimizationProblem``.

    Returns:
        JSON-serializable dict with ``sense``, ``n_vars``, ``constant``,
        ``linear``, ``quadratic``, and ``higher_order``.

    Raises:
        TypeError / ValueError: If ``op`` fails unconstrained binary validation.
    """
    validate_optimization_problem(op)
    normalized = _normalize_binary_polynomial(op)
    objective = normalized.objective
    sense = _objective_sense(normalized)

    linear = {str(int(i)): float(c) for i, c in objective.linear.to_dict().items()}
    quadratic = {_index_key((int(i), int(j))): float(c) for (i, j), c in objective.quadratic.to_dict().items()}
    higher_order: dict[str, dict[str, float]] = {}
    for order, expr in (objective.higher_order or {}).items():
        mapping = expr.to_dict() if hasattr(expr, "to_dict") else dict(expr)
        higher_order[str(int(order))] = {_index_key(tuple(int(i) for i in key)): float(coeff) for key, coeff in mapping.items()}

    return {
        "sense": sense,
        "n_vars": int(normalized.get_num_vars()),
        "constant": float(getattr(objective, "constant", 0.0)),
        "linear": linear,
        "quadratic": quadratic,
        "higher_order": higher_order,
    }


def deserialize_optimization_problem(payload: dict[str, Any]) -> OptimizationProblem:
    """Rebuild an ``OptimizationProblem`` from a ``polynomial_problem`` payload.

    Args:
        payload: Dict produced by ``serialize_optimization_problem`` (or equivalent).

    Returns:
        Validated, normalized unconstrained binary ``OptimizationProblem``.

    Raises:
        TypeError: If ``payload`` is not a mapping.
        ValueError: If required fields are missing/invalid, or validation fails.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"polynomial_problem must be a dict, got {type(payload)!r}.")
    try:
        n_vars = int(payload["n_vars"])
        sense = str(payload.get("sense", "min")).lower()
        constant = float(payload.get("constant", 0.0))
        linear_raw = payload.get("linear") or {}
        quadratic_raw = payload.get("quadratic") or {}
        higher_order_raw = payload.get("higher_order") or {}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid polynomial_problem payload.") from exc

    if n_vars < 1:
        raise ValueError("polynomial_problem n_vars must be >= 1.")
    if sense not in ("min", "max"):
        raise ValueError("polynomial_problem sense must be 'min' or 'max'.")

    def _bound(indices: tuple[int, ...]) -> tuple[int, ...]:
        for idx in indices:
            if idx < 0 or idx >= n_vars:
                raise ValueError(f"polynomial_problem index {idx} is outside [0, {n_vars}).")
        return indices

    try:
        linear = {_bound((int(k),))[0]: float(v) for k, v in dict(linear_raw).items()}
        quadratic = {_bound(_parse_index_key(str(k))): float(v) for k, v in dict(quadratic_raw).items()}
        higher_order: dict[int, dict[tuple[int, ...], float]] = {}
        for order, terms in dict(higher_order_raw).items():
            order_i = int(order)
            if order_i < 3:
                raise ValueError(f"polynomial_problem higher_order key {order_i} must be >= 3.")
            parsed_terms = {}
            for key, coeff in dict(terms).items():
                idxs = _bound(_parse_index_key(str(key)))
                if len(idxs) != order_i:
                    raise ValueError(f"polynomial_problem higher_order key {key!r} must have {order_i} indices.")
                parsed_terms[idxs] = float(coeff)
            higher_order[order_i] = parsed_terms
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ValueError) and "polynomial_problem" in str(exc):
            raise
        raise ValueError("Invalid polynomial_problem payload.") from exc

    op = OptimizationProblem("polynomial")
    op.binary_var_list(n_vars)
    kwargs: dict[str, Any] = {
        "constant": constant,
        "linear": linear or None,
        "quadratic": quadratic or None,
    }
    if higher_order:
        kwargs["higher_order"] = higher_order
    if sense == "max":
        op.maximize(**kwargs)
    else:
        op.minimize(**kwargs)
    validate_optimization_problem(op)
    return _normalize_binary_polynomial(op)


def evaluate_problem_cost(problem: Any, bitstring: str) -> float:
    """Evaluate cost for a Qiskit little-endian bitstring on OP or QUBO."""
    if isinstance(problem, OptimizationProblem):
        n = int(problem.get_num_vars())
        if len(bitstring) != n or any(ch not in "01" for ch in bitstring):
            raise ValueError(f"Bitstring must have length {n} and only contain '0'/'1'.")
        x = [int(ch) for ch in bitstring[::-1]]
        return float(problem.objective.evaluate(x))
    if hasattr(problem, "cost"):
        return float(problem.cost(bitstring))
    raise TypeError(f"Cannot evaluate cost for problem type {type(problem)!r}.")


def uses_polynomial_wire(problem: Any) -> bool:
    """True for supported ``OptimizationProblem`` inputs, regardless of degree."""
    from .qubo import QUBO

    if isinstance(problem, QUBO):
        return False
    if isinstance(problem, OptimizationProblem):
        return True
    return False


def _as_qubo(problem: Any, *, warn_deprecated: bool = True) -> "QUBO":
    """Convert a public problem to the internal LP adapter for the quadratic kernel."""
    from .qubo import QUBO, QUBO_CLASS_DEPRECATION_MSG

    if isinstance(problem, QUBO):
        if warn_deprecated and not getattr(problem, "_haiqu_qubo_deprecated_warned", False):
            warnings.warn(QUBO_CLASS_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)
            problem._haiqu_qubo_deprecated_warned = True
        return problem
    if isinstance(problem, OptimizationProblem):
        adapter = QUBO._from_optimization_problem(problem)
        adapter._haiqu_qubo_deprecated_warned = True
        return adapter
    raise TypeError(
        "problem must be qiskit_addon_opt_mapper.problems.OptimizationProblem "
        f"(or the deprecated haiqu.sdk.optimization.QUBO), got {type(problem)!r}."
    )


ProblemLike = Union[OptimizationProblem, Any]

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
]
