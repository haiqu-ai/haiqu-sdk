"""
Fermionic Hamiltonian helper functions for SKQD.

Standalone functions for building one-body (h1e) + two-body (h2e) tensors
and performing basis rotations. No model classes — just data in, data out.
"""

import numpy as np


def hubbard_hamiltonian(
    norb: int,
    t: float,
    U: float,
    periodic: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Build 1D Fermi-Hubbard Hamiltonian tensors in site basis.

    .. code-block:: text

        H = -t * sum_{<i,j>,sigma} (c^dag_i c_j + h.c.) + U * sum_i n_up_i n_down_i

    Args:
        norb: Number of spatial orbitals (lattice sites). Must be >= 2.
        t: Hopping parameter (kinetic energy).
        U: On-site interaction strength (potential energy).
        periodic: Whether to use periodic boundary conditions.

    Returns:
        (h1e, h2e) — one-body tensor shape (norb, norb) and
        two-body tensor shape (norb, norb, norb, norb).
    """
    if norb < 2:
        raise ValueError(f"norb must be >= 2, got {norb}")

    h1e = np.zeros((norb, norb))
    for i in range(norb - 1):
        h1e[i, i + 1] = h1e[i + 1, i] = -t
    if periodic and norb > 2:
        h1e[0, norb - 1] = h1e[norb - 1, 0] = -t

    h2e = np.zeros((norb, norb, norb, norb))
    for i in range(norb):
        h2e[i, i, i, i] = U

    return h1e, h2e


def siam_hamiltonian(
    norb: int,
    t: float,
    U: float,
    V: float,
    mu: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build single-impurity Anderson model Hamiltonian tensors in site basis.

    .. code-block:: text

        H = -t * sum_{<i,j> in bath, sigma} (c^dag_i c_j + h.c.)
          - V * sum_sigma (c^dag_d c_1 + h.c.)
          + mu * n_d
          + U * n_d_up * n_d_down

    The impurity is placed on site 0.

    Args:
        norb: Number of spatial orbitals (impurity + bath sites). Must be >= 2.
        t: Hopping parameter between bath sites.
        U: On-site interaction strength at the impurity.
        V: Impurity-bath hybridization.
        mu: Chemical potential at the impurity.

    Returns:
        (h1e, h2e) — one-body tensor shape (norb, norb) and
        two-body tensor shape (norb, norb, norb, norb).
    """
    if norb < 2:
        raise ValueError(f"norb must be >= 2, got {norb}")

    impurity_orb = 0

    h1e = np.zeros((norb, norb))
    np.fill_diagonal(h1e[:, 1:], -t)
    np.fill_diagonal(h1e[1:, :], -t)
    h1e[impurity_orb, impurity_orb + 1] = -V
    h1e[impurity_orb + 1, impurity_orb] = -V
    h1e[impurity_orb, impurity_orb] = mu

    h2e = np.zeros((norb, norb, norb, norb))
    h2e[impurity_orb, impurity_orb, impurity_orb, impurity_orb] = U

    return h1e, h2e


def get_orbital_rotation(norb: int) -> np.ndarray:
    """Build the orbital rotation matrix for the SIAM model.

    Constructs a unitary matrix C that transforms Hamiltonian tensors from
    site basis to momentum basis via ``rotate_basis(h1e, h2e, C^dagger)``.

    The ground state of the Anderson impurity model is significantly sparser
    in momentum basis, making SQD sampling more effective. The rotation:

    1. Diagonalizes the bath hopping chain (sites 1..norb-1) to obtain
       momentum eigenstates of the non-interacting bath.
    2. Leaves the impurity orbital (site 0) unchanged.
    3. Permutes the impurity to the center of the orbital ordering so
       that the on-site interaction sits at index ``(norb - 1) // 2``
       in the new basis.

    Args:
        norb: Number of spatial orbitals (impurity + bath sites).

    Returns:
        Unitary rotation matrix C, shape (norb, norb). Use as::

            h1e_mom, h2e_mom = rotate_basis(h1e, h2e, C.T.conj())
    """
    n_bath = norb - 1

    hopping_matrix = np.zeros((n_bath, n_bath))
    np.fill_diagonal(hopping_matrix[:, 1:], -1)
    np.fill_diagonal(hopping_matrix[1:, :], -1)
    _, vecs = np.linalg.eigh(hopping_matrix)

    orbital_rotation = np.zeros((norb, norb))
    orbital_rotation[0, 0] = 1
    orbital_rotation[1:, 1:] = vecs

    new_index = n_bath // 2
    perm = np.r_[1 : (new_index + 1), 0, (new_index + 1) : norb]  # noqa: E203
    orbital_rotation = orbital_rotation[:, perm]

    return orbital_rotation


def rotate_basis(
    h1e: np.ndarray,
    h2e: np.ndarray,
    orbital_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate the orbital basis of one- and two-body Hamiltonian tensors.

    Applies the unitary transformation::

        h1e'[A,B]       = sum_{a,b} U[A,a] * h1e[a,b] * U*[B,b]
        h2e'[A,B,C,D]   = sum_{a,b,c,d} U[A,a] * U*[B,b] * U[C,c] * U*[D,d] * h2e[a,b,c,d]

    where U = ``orbital_rotation`` and U* is its complex conjugate.

    Args:
        h1e: One-body tensor, shape (norb, norb).
        h2e: Two-body tensor, shape (norb, norb, norb, norb).
        orbital_rotation: Unitary matrix U, shape (norb, norb).

    Returns:
        (h1e_rotated, h2e_rotated) in the new basis.
    """
    h1e_rotated = np.einsum(
        "ab,Aa,Bb->AB",
        h1e,
        orbital_rotation,
        orbital_rotation.conj(),
        optimize="greedy",
    )
    h2e_rotated = np.einsum(
        "abcd,Aa,Bb,Cc,Dd->ABCD",
        h2e,
        orbital_rotation,
        orbital_rotation.conj(),
        orbital_rotation,
        orbital_rotation.conj(),
        optimize="greedy",
    )
    return h1e_rotated, h2e_rotated
