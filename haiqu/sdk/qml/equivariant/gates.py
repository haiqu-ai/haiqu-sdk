"""Haiqu SDK QML: SU(2)-equivariant 2- and 3-qubit gates.

Building blocks for symmetry-preserving variational ansatze. The 2-qubit
gate is the unique SU(2)-equivariant gate on 2 qubits (up to global phase);
the 3-qubit gate is the exact equivariant gate on 3 qubits.
"""

from __future__ import annotations

import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import CUGate, PhaseGate, UnitaryGate


def su2_equivariant_2_qubit_gate(theta) -> QuantumCircuit:
    """The 2-qubit SU(2)-equivariant gate.

    Under total spin, two qubits split into a 1-dimensional spin-0 "singlet"
    subspace (the antisymmetric state ``(|01> - |10>)/sqrt(2)``) and a
    3-dimensional spin-1 "triplet" subspace (the symmetric states). This gate
    phases the singlet by ``exp(i*theta)`` and acts as identity on the
    triplet. It is the unique 2-qubit SU(2)-equivariant gate (up to global
    phase) and the building block for the n-qubit ansatz.

    Hardware form: ``cx . crx(theta) . p(theta/2) . cx``.

    Args:
        theta: Angle. Accepts ``float`` or a qiskit ``Parameter`` /
            ``ParameterExpression``; the gate is equivariant for any value.

    Returns:
        A 2-qubit ``QuantumCircuit`` implementing the gate.

    Example:
        >>> from haiqu.sdk.qml.equivariant import su2_equivariant_2_qubit_gate
        >>> gate = su2_equivariant_2_qubit_gate(0.7)
        >>> gate.num_qubits
        2
    """
    qc = QuantumCircuit(2, name="su2_2q")
    qc.cx(0, 1)
    qc.crx(theta, 1, 0)
    qc.p(theta / 2, 1)
    qc.cx(0, 1)
    return qc


# 3-qubit Schur (spin-coupling) basis change, in qiskit little-endian ordering.
_SCHUR3 = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0, 0, 0, 0, 0],
        [0, 0, 0, 1 / np.sqrt(3), 0, 1 / np.sqrt(3), 1 / np.sqrt(3), 0],
        [0, -1 / np.sqrt(6), -1 / np.sqrt(6), 0, np.sqrt(2) / np.sqrt(3), 0, 0, 0],
        [0, 1 / np.sqrt(3), 1 / np.sqrt(3), 0, 1 / np.sqrt(3), 0, 0, 0],
        [0, 0, 0, 0, 0, 1 / np.sqrt(2), -1 / np.sqrt(2), 0],
        [0, 0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, -np.sqrt(2) / np.sqrt(3), 0, 1 / np.sqrt(6), 1 / np.sqrt(6), 0],
    ],
    dtype=complex,
)


def su2_equivariant_3_qubit_gate(theta0, theta1, theta2, theta3) -> QuantumCircuit:
    """The exact 3-qubit SU(2)-equivariant gate.

    The 3-qubit Hilbert space decomposes as spin-3/2 (dim 4, multiplicity 1)
    plus two copies of spin-1/2 (dim 2, multiplicity 2). The equivariant gate
    is identity on the spin-3/2 sector and applies a generic ``U_2 (x) I_2``
    that mixes the two spin-1/2 copies while leaving the magnetic label
    intact. The four angles parametrize ``U_2``.

    Constructed as ``S_3 . diag(I_4, U_2(theta) (x) I_2) . S_3^dagger`` where
    ``S_3`` is the 3-qubit Schur transform. The four angles are the Euler
    angles of ``U_2`` in qiskit's ``CUGate(theta, phi, lam, gamma)``
    convention (a controlled ``U(theta, phi, lam)`` with global phase
    ``gamma``):

    Args:
        theta0: Global phase applied to the ``U_2`` block (``PhaseGate``).
        theta1: The ``lam`` Euler angle of ``U_2``.
        theta2: The ``theta`` (rotation) Euler angle of ``U_2``.
        theta3: The ``phi`` Euler angle of ``U_2``.

    Returns:
        A 3-qubit ``QuantumCircuit`` implementing the gate.
    """
    schur = UnitaryGate(_SCHUR3, label="S3")
    qc = QuantumCircuit(3, name="su2_3q")
    qc.append(schur, [0, 1, 2])
    qc.append(CUGate(theta2, theta3, theta1, -(theta1 + theta3) / 2), [0, 1])
    qc.append(PhaseGate(theta0), [0])
    qc.append(schur.inverse(), [0, 1, 2])
    return qc
