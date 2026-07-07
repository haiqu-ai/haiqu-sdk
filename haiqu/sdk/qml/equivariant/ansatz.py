"""Haiqu SDK QML: SU(2)-equivariant parametrized ansatz.

Builds parametrized ``QuantumCircuit`` ansatze from the
``su2_equivariant_2_qubit_gate`` primitive. Equivariance is structural: for any
parameter binding, the result commutes with the global spin generators
``S_x``, ``S_y``, ``S_z``.
"""

from __future__ import annotations

from typing import List, Union

from qiskit.circuit import ParameterVector, QuantumCircuit

from .gates import su2_equivariant_2_qubit_gate

Pattern = List[List[int]]


def brickwork_pattern(num_qubits: int, num_layers: int) -> Pattern:
    """Nearest-neighbour brickwork pattern: even bonds, then odd bonds, repeated.

    Args:
        num_qubits: Number of qubits.
        num_layers: Number of brickwork layers.

    Returns:
        A list of ``[i, j]`` qubit pairs in application order.

    Example:
        >>> brickwork_pattern(4, 1)
        [[0, 1], [2, 3], [1, 2]]
    """
    even = [[i, i + 1] for i in range(0, num_qubits - 1, 2)]
    odd = [[i, i + 1] for i in range(1, num_qubits - 1, 2)]
    layer = even + odd
    return layer * num_layers


def su2_equivariant_ansatz(
    num_qubits: int,
    layout: Union[str, Pattern] = "brickwork",
    num_layers: int = 1,
    name: str = "theta",
) -> QuantumCircuit:
    """Build a parametrized SU(2)-equivariant ansatz.

    Returns a ``QuantumCircuit`` with a ``ParameterVector`` of free angles,
    one per 2-qubit ``su2`` gate. Equivariance is a structural property of
    the building block and holds for any parameter binding the optimiser
    ever tries: ``su2(theta) = exp(i*theta*P_ij)`` where
    ``P_ij = (I - SWAP_ij) / 2`` is the pair-singlet projector, and
    ``P_ij`` commutes with the global ``S_x``, ``S_y``, ``S_z``. Products
    of equivariant unitaries are equivariant, so the whole parametrized
    circuit lies inside the equivariant subgroup of unitaries.

    Contrast with Qiskit's ``EfficientSU2``: that ansatz uses single-qubit
    Pauli rotations, which do NOT commute with the global spin operators,
    so it is not equivariant for any non-trivial parameter binding.

    Args:
        num_qubits: Number of qubits.
        layout: ``"brickwork"`` (nearest-neighbour even-then-odd; the
            default), ``"linear"`` (chain of adjacent pairs), or a list of
            ``[i, j]`` qubit-pair lists for a custom layout.
        num_layers: Repetitions of the chosen layout pattern. Defaults to 1.
        name: ``ParameterVector`` name (default ``"theta"``).

    Returns:
        A parametrized ``QuantumCircuit`` over ``num_qubits`` qubits. Free
        parameters are accessible via ``circuit.parameters``.

    Raises:
        ValueError: If ``layout`` is not a recognised string or a list of
            qubit pairs.

    Example:
        >>> from haiqu.sdk.qml.equivariant import su2_equivariant_ansatz
        >>> ansatz = su2_equivariant_ansatz(num_qubits=4, num_layers=2)
        >>> len(ansatz.parameters)
        6
    """
    if layout == "brickwork":
        pattern = brickwork_pattern(num_qubits, num_layers)
    elif layout == "linear":
        chain = [[i, i + 1] for i in range(num_qubits - 1)]
        pattern = chain * num_layers
    elif isinstance(layout, (list, tuple)):
        pattern = [list(pair) for _ in range(num_layers) for pair in layout]
    else:
        raise ValueError(f"unknown layout: {layout!r}")

    params = ParameterVector(name, len(pattern))
    qc = QuantumCircuit(num_qubits, name="su2_ansatz")
    for theta, wires in zip(params, pattern):
        qc.compose(su2_equivariant_2_qubit_gate(theta), qubits=wires, inplace=True)
    return qc
