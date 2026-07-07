"""
Haiqu SDK: Quantum Machine Learning (QML) module.

Provides support for variational quantum optimization and
SU(2)-equivariant ansatze (under :mod:`haiqu.sdk.qml.equivariant`, with
the public surface re-exported here for convenience).
"""

from .compression_options import CompressionOptions
from .equivariant import (
    brickwork_pattern,
    is_su2_equivariant,
    spin_generators,
    su2_equivariant_2_qubit_gate,
    su2_equivariant_3_qubit_gate,
    su2_equivariant_ansatz,
    total_spin_ops,
)
from .optimizer import NFTOptimizerOptions, OptimizerOptions, ScipyOptimizerOptions
from .problem import VariationalProblem, NonlinearVariationalProblem

__all__ = [
    # Variational optimization
    "CompressionOptions",
    "VariationalProblem",
    "NonlinearVariationalProblem",
    "OptimizerOptions",
    "NFTOptimizerOptions",
    "ScipyOptimizerOptions",
    # SU(2)-equivariant gates and ansatze
    "brickwork_pattern",
    "is_su2_equivariant",
    "spin_generators",
    "su2_equivariant_2_qubit_gate",
    "su2_equivariant_3_qubit_gate",
    "su2_equivariant_ansatz",
    "total_spin_ops",
]
