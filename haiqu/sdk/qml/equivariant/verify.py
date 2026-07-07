"""Haiqu SDK QML: SU(2)-equivariance verifier."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .spin import spin_generators


def _commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return A @ B - B @ A


def is_su2_equivariant(U, tol: float = 1e-8) -> Tuple[bool, float]:
    """Check whether a unitary is SU(2)-equivariant.

    A gate ``U`` on n qubits is SU(2)-equivariant iff it commutes with the
    three global spin generators ``S_x``, ``S_y``, ``S_z``. Commuting with
    every generator is necessary AND sufficient for commuting with the whole
    SU(2) group, and avoids forming the total-spin Casimir
    ``S^2 = S_x^2 + S_y^2 + S_z^2`` (each ``S_a`` is a sparse sum of n Paulis).

    Checking only ``[U, S^2] = [U, S_z] = 0`` (with ``S^2`` the Casimir above)
    is strictly weaker: for example ``exp(i*phi*S_z)`` commutes with both yet
    is NOT equivariant.

    Args:
        U: ``2^n`` by ``2^n`` unitary as an array-like.
        tol: Threshold on the largest commutator entry.

    Returns:
        Tuple ``(ok, violation)`` where
        ``ok = (max_a |[U, S_a]| < tol)`` and ``violation`` is the maximum
        commutator entry over ``a`` in ``{x, y, z}``.

    Raises:
        ValueError: If the matrix shape is not ``(2^n, 2^n)``.
    """
    U = np.asarray(U, dtype=complex)
    d = U.shape[0]
    n = int(round(np.log2(d)))
    if 2**n != d or U.shape != (d, d):
        raise ValueError(f"matrix shape {U.shape} is not (2^n, 2^n)")
    viol = max(abs(_commutator(U, S)).max() for S in spin_generators(n))
    return bool(viol < tol), float(viol)
