"""Haiqu SDK QML: spin operator utilities.

Dense SU(2) generators (S_x, S_y, S_z) and the total-spin Casimir
S^2 = S_x^2 + S_y^2 + S_z^2 for n qubits, in the qiskit qubit convention.
Useful for building custom symmetry-preserving Hamiltonians or equivariance
checks.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from qiskit.quantum_info import SparsePauliOp


def _pauli_sum(n: int, axis: str) -> np.ndarray:
    """Return the dense matrix of ``S_axis = (1/2) sum_i sigma_axis^(i)``."""
    terms = []
    for i in range(n):
        lbl = ["I"] * n
        # qiskit label convention: qubit 0 is on the right.
        lbl[n - 1 - i] = axis
        terms.append(("".join(lbl), 0.5))
    return SparsePauliOp.from_list(terms).to_matrix()


def spin_generators(n: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dense SU(2) generators ``(S_x, S_y, S_z)`` for n qubits.

    Args:
        n: Number of qubits.

    Returns:
        A tuple ``(S_x, S_y, S_z)`` of dense ``(2**n, 2**n)`` complex matrices.
    """
    return tuple(_pauli_sum(n, a) for a in "XYZ")


def total_spin_ops(n: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return dense matrices ``(S^2, S_z)`` for n qubits.

    Args:
        n: Number of qubits.

    Returns:
        Tuple of dense ``(2**n, 2**n)`` matrices for the total-spin Casimir
        operator ``S^2 = S_x^2 + S_y^2 + S_z^2`` and the z-component ``S_z``.
    """
    Sx, Sy, Sz = spin_generators(n)
    S2 = Sx @ Sx + Sy @ Sy + Sz @ Sz
    return S2, Sz
