"""
Haiqu SDK QML: Variational Problem definition.
"""

from __future__ import annotations

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp


class VariationalProblem:
    """A variational quantum optimization problem definition.

    Bundles a parameterized ansatz circuit with an observable to minimize.

    Args:
        ansatz: Parameterized quantum circuit.
        observable: The observable as a SparsePauliOp.

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
