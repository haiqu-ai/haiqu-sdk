"""
haiqu.sdk.skqd - Sample-based Krylov Quantum Diagonalization.

Helper functions for building Hamiltonian tensors, performing basis
rotations, and generating Krylov subspace circuits. Postprocessing
(SQD diagonalization) runs server-side via ``haiqu.postprocess_skqd``.
"""

from .hamiltonian import (
    hubbard_hamiltonian,
    siam_hamiltonian,
    get_orbital_rotation,
    rotate_basis,
)
from .config import SKQDOptions
from .circuits_hubbard import build_hubbard_site_basis_krylov_circuits
from .circuits_siam import build_siam_momentum_basis_krylov_circuits

__all__ = [
    "hubbard_hamiltonian",
    "siam_hamiltonian",
    "get_orbital_rotation",
    "rotate_basis",
    "SKQDOptions",
    "build_hubbard_site_basis_krylov_circuits",
    "build_siam_momentum_basis_krylov_circuits",
]
