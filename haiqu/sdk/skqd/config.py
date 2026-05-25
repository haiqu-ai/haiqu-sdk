"""
SKQD postprocessing options.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(kw_only=True)
class SKQDOptions:
    """Options for SQD postprocessing (server-side diagonalization).

    These parameters control the classical SQD algorithm, not circuit
    generation. Circuit parameters (krylov_dim, dt) are passed directly
    to the circuit builder functions.

    Args:
        samples_per_batch: Number of bitstring samples per SQD batch.
        num_batches: Number of subsampling batches for SQD.
        max_iterations: Maximum number of SQD self-consistent iterations.
        symmetrize_spin: Whether to enforce spin symmetry in SQD.
        configuration_recovery: Whether to apply configuration recovery to
            refine noisy bitstrings using orbital occupancy information. See
            [Configuration recovery](configuration_recovery.md) for what this does
            and when to use it.
        seed: Random seed for reproducibility.
    """

    samples_per_batch: int = 100
    num_batches: int = 5
    max_iterations: int = 15
    symmetrize_spin: bool = True
    configuration_recovery: bool = False
    seed: Optional[int] = None
