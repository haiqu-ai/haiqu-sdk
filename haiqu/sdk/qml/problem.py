"""
Haiqu SDK QML: Variational Problem definition.

Observable term strings in :class:`NonlinearVariationalProblem` use Qiskit's
reversed-order (little-endian) convention: the rightmost character acts on qubit 0.
"""

from __future__ import annotations

import sympy
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from ..utils import sparse_op_to_tuple

# Characters allowed in an observable term string: the Paulis plus the
# single-qubit computational-basis projectors |0><0| ("0") and |1><1| ("1").
_ALLOWED_TERM_CHARS = frozenset("IXYZ01")

_TERM_STRING_ORDER_NOTE = (
    "Term strings use Qiskit's reversed-order (little-endian) convention: the "
    "**rightmost** character acts on **qubit 0** (``q_0``), the leftmost on qubit "
    "``n - 1``. This applies uniformly to Pauli symbols (``I``, ``X``, ``Y``, ``Z``) "
    'and projector symbols (``0``, ``1``). For example, on 2 qubits, ``"IZ"`` is '
    '``Z`` on ``q_0`` and ``"ZI"`` is ``Z`` on ``q_1``; ``"0I"`` is '
    '``|0⟩⟨0|`` on ``q_1`` and ``I`` on ``q_0``, while ``"I0"`` is '
    "``|0⟩⟨0|`` on ``q_0`` and ``I`` on ``q_1``. "
    "``SparsePauliOp`` observables use the same label convention (this matches amplitude / "
    "bitstring little-endian indexing: rightmost position = ``q_0``). Projector terms "
    "must be given as ``(term_string, coefficient)`` pairs — ``SparsePauliOp`` "
    "cannot represent ``0``/``1`` symbols."
)


class VariationalProblem:
    """A variational quantum optimization problem definition.

    Bundles a parameterized ansatz circuit with an observable to minimize.

    Args:
        ansatz: Parameterized quantum circuit.
        observable: The observable as a SparsePauliOp. Pauli labels follow Qiskit's
            reversed-order convention (rightmost character = qubit 0), the same as
            :class:`NonlinearVariationalProblem` term strings.

    Raises:
        TypeError: If inputs are wrong types.
        ValueError: If ansatz is not parameterized.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> from qiskit.circuit import Parameter
        >>> from qiskit.quantum_info import SparsePauliOp
        >>> from haiqu.sdk.qml import VariationalProblem
        >>> theta = Parameter('θ')
        >>> ansatz = QuantumCircuit(2)
        >>> ansatz.ry(theta, 0)
        >>> ansatz.cx(0, 1)
        >>> obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5)])
        >>> problem = VariationalProblem(ansatz, obs)

    See Also:
        :meth:`haiqu.sdk.quantum_haiqu.Haiqu.variational_optimization`: Submit problem to Haiqu cloud.
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        observable: SparsePauliOp,
    ):
        if not isinstance(ansatz, QuantumCircuit):
            raise TypeError("ansatz must be a QuantumCircuit.")

        if not isinstance(observable, SparsePauliOp):
            raise TypeError("observable must be a SparsePauliOp.")

        if not ansatz.num_parameters:
            raise ValueError("ansatz must be parameterized (have unbound Parameters).")

        self._ansatz = ansatz
        self._observable = observable

    @property
    def ansatz(self) -> QuantumCircuit:
        """The variational ansatz circuit."""
        return self._ansatz

    @property
    def observable(self) -> SparsePauliOp:
        """The observable to minimize."""
        return self._observable

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the problem."""
        return self._ansatz.num_qubits

    @property
    def num_parameters(self) -> int:
        """Number of variational parameters in the ansatz."""
        return self._ansatz.num_parameters


class NonlinearVariationalProblem:
    """A non-linear variational quantum optimization problem definition.

    Generalizes :class:`VariationalProblem`: instead of minimizing the expectation value
    of a single observable, it minimizes ``loss(<O_1>, <O_2>, ...)`` where ``loss`` is a
    sympy expression and each free symbol maps to its own observable.

    Observable terms may contain the single-qubit computational-basis projector symbols
    ``"0"`` (``|0><0|``) and ``"1"`` (``|1><1|``) in addition to the Pauli characters
    ``I``/``X``/``Y``/``Z``.

    {_TERM_STRING_ORDER_NOTE}

    Args:
        ansatz: Parameterized quantum circuit.
        loss: The non-linear objective, as a sympy expression or a string that sympifies to
            one (e.g. ``"1 - x/y"``). It is sympified and simplified on construction.
        observables: Mapping from each free symbol of ``loss`` (given as a ``str`` or a
            ``sympy.Symbol``) to its observable. Each observable may be a ``SparsePauliOp``
            or a list of ``(term_string, coefficient)`` pairs whose term strings are over the
            alphabet ``{I, X, Y, Z, 0, 1}``. Each ``term_string`` must have length
            ``ansatz.num_qubits`` and follow the Qiskit term-string order above.

    Raises:
        TypeError: If inputs are wrong types.
        ValueError: If the ansatz is not parameterized, ``loss`` does not parse, ``loss``
            contains the imaginary unit, the ``loss`` free symbols do not match the
            observable keys, or an observable term is malformed.

    Example:
        >>> from qiskit import QuantumCircuit
        >>> from qiskit.circuit import Parameter
        >>> from haiqu.sdk.qml import NonlinearVariationalProblem
        >>> theta = Parameter('θ')
        >>> ansatz = QuantumCircuit(2)
        >>> ansatz.ry(theta, 0)
        >>> ansatz.cx(0, 1)
        >>> problem = NonlinearVariationalProblem(
        ...     ansatz,
        ...     "1 - x/y",
        ...     {"x": [("ZI", 1.0)], "y": [("0I", 0.5), ("1Z", -0.5)]},
        ... )

    See Also:
        :class:`VariationalProblem`: The linear, single-observable problem.
    """

    def __init__(
        self,
        ansatz: QuantumCircuit,
        loss: sympy.Expr | str,
        observables: dict,
    ):
        if not isinstance(ansatz, QuantumCircuit):
            raise TypeError("ansatz must be a QuantumCircuit.")

        if not ansatz.num_parameters:
            raise ValueError("ansatz must be parameterized (have unbound Parameters).")

        expr = self._normalize_loss(loss)
        normalized = self._normalize_observables(observables, ansatz.num_qubits)

        symbol_names = {str(symbol) for symbol in expr.free_symbols}
        keys = set(normalized)
        if symbol_names != keys:
            missing = symbol_names - keys
            extra = keys - symbol_names
            raise ValueError(
                f"loss free symbols must match observable keys. Simplified loss is {expr!r} "
                f"(missing observables for {sorted(missing)}; unused observable keys {sorted(extra)}). "
                "Unused keys may have cancelled out during simplification (e.g. 'x*y/y' simplifies to 'x')."
            )

        self._ansatz = ansatz
        self._loss = expr
        self._observables = normalized

    @staticmethod
    def _normalize_loss(loss: sympy.Expr | str) -> sympy.Expr:
        """Sympify (parse if a string) and simplify ``loss``, rejecting the imaginary unit."""
        try:
            expr = sympy.sympify(loss)
        except (sympy.SympifyError, SyntaxError, TypeError) as exc:
            raise ValueError(f"loss could not be parsed as a sympy expression: {exc}") from exc

        expr = sympy.simplify(expr)

        if expr.has(sympy.I):
            raise ValueError("loss must not contain the imaginary unit 'I'.")

        return expr

    @staticmethod
    def _normalize_observables(observables: dict, num_qubits: int) -> dict[str, list[tuple[str, float]]]:
        """Normalize observable keys to symbol names and values to validated ``(str, float)`` pairs."""
        if not isinstance(observables, dict):
            raise TypeError("observables must be a dict mapping symbol names to observables.")

        normalized: dict[str, list[tuple[str, float]]] = {}
        for key, value in observables.items():
            if isinstance(key, sympy.Symbol):
                name = key.name
            elif isinstance(key, str):
                name = key
            else:
                raise TypeError("observable keys must be str or sympy.Symbol.")

            if name in normalized:
                raise ValueError(f"duplicate observable key {name!r}.")

            normalized[name] = NonlinearVariationalProblem._normalize_terms(name, value, num_qubits)

        return normalized

    @staticmethod
    def _normalize_terms(name: str, value, num_qubits: int) -> list[tuple[str, float]]:
        """Convert one observable to a validated list of ``(term_string, real_coefficient)`` pairs."""
        if isinstance(value, SparsePauliOp):
            # sparse_op_to_tuple asserts real coefficients (Hermiticity) rather than silently
            # dropping any imaginary part.
            labels, coeffs = sparse_op_to_tuple(value)
            pairs = list(zip(labels, coeffs))
        elif isinstance(value, (list, tuple)):
            pairs = []
            for item in value:
                if not (isinstance(item, (list, tuple)) and len(item) == 2):
                    raise ValueError(f"observable {name!r} must be a list of (term_string, coefficient) pairs.")
                term, coeff = item
                if not isinstance(term, str):
                    raise TypeError(f"observable {name!r} term strings must be str.")
                if not np.issubdtype(type(coeff), np.number):
                    raise ValueError(
                        f"observable {name!r} coefficient for term {term!r} must be a number, " f"got {type(coeff).__name__}."
                    )
                if not np.isclose(np.imag(coeff), 0.0):
                    raise ValueError(
                        f"observable {name!r} must be a valid Hermitian operator with real coefficients. "
                        f"Found term {term!r} with complex coefficient {coeff}."
                    )
                pairs.append((term, np.real(coeff)))
        else:
            raise TypeError(f"observable {name!r} must be a SparsePauliOp or a list of (term_string, coefficient) pairs.")

        if not pairs:
            raise ValueError(f"observable {name!r} must have at least one term.")

        for term, _ in pairs:
            if len(term) != num_qubits:
                raise ValueError(
                    f"observable {name!r} term {term!r} has length {len(term)}, expected {num_qubits} (ansatz qubits)."
                )
            invalid = set(term) - _ALLOWED_TERM_CHARS
            if invalid:
                raise ValueError(
                    f"observable {name!r} term {term!r} contains invalid characters {sorted(invalid)}; "
                    "allowed: I, X, Y, Z, 0, 1."
                )

        return pairs

    @property
    def ansatz(self) -> QuantumCircuit:
        """The variational ansatz circuit."""
        return self._ansatz

    @property
    def loss(self) -> sympy.Expr:
        """The non-linear objective as a simplified sympy expression."""
        return self._loss

    @property
    def observables(self) -> dict[str, list[tuple[str, float]]]:
        """Mapping from symbol name to its observable as ``(term_string, coefficient)`` pairs."""
        return self._observables

    @property
    def num_qubits(self) -> int:
        """Number of qubits in the problem."""
        return self._ansatz.num_qubits

    @property
    def num_parameters(self) -> int:
        """Number of variational parameters in the ansatz."""
        return self._ansatz.num_parameters


NonlinearVariationalProblem.__doc__ = NonlinearVariationalProblem.__doc__.replace(
    "{_TERM_STRING_ORDER_NOTE}", _TERM_STRING_ORDER_NOTE
)
