"""Haiqu SDK QML: SU(2)-equivariant gates and ansatze.

Symmetry-preserving variational primitives for Heisenberg-style Hamiltonians
(Kagome lattice and other spin-system targets).

Public API:

- :func:`su2_equivariant_ansatz`: parametrized ansatz, the VQE / QML
  headline.
- :func:`su2_equivariant_2_qubit_gate`: 2-qubit equivariant primitive (1 angle).
- :func:`su2_equivariant_3_qubit_gate`: exact 3-qubit equivariant gate (4 angles).
- :func:`brickwork_pattern`: nearest-neighbour layout helper.
- :func:`is_su2_equivariant`: equivariance check (three-generator commutators).
- :func:`spin_generators`, :func:`total_spin_ops`: dense SU(2) generators and
  total-spin Casimir, useful for building custom symmetry-preserving
  Hamiltonians or checks.

Compressing an equivariant gate into a shallow 2-qubit ``su2`` brick is a
server-side job, submitted via ``haiqu.su2_equivariant_compilation``.
"""

from .ansatz import brickwork_pattern, su2_equivariant_ansatz
from .gates import su2_equivariant_2_qubit_gate, su2_equivariant_3_qubit_gate
from .spin import spin_generators, total_spin_ops
from .verify import is_su2_equivariant

__all__ = [
    "brickwork_pattern",
    "is_su2_equivariant",
    "spin_generators",
    "su2_equivariant_2_qubit_gate",
    "su2_equivariant_3_qubit_gate",
    "su2_equivariant_ansatz",
    "total_spin_ops",
]
