"""
Haiqu SDK QML: Optimizer configuration classes.
"""

from __future__ import annotations

from pydantic import BaseModel


class OptimizerOptions(BaseModel):
    """Base class for optimizer configuration.

    Subclass this to create configuration for specific optimizers.
    Do not instantiate directly; use a subclass like NFTOptimizerOptions.
    """

    def __init__(self, **data):
        if type(self) is OptimizerOptions:
            raise TypeError("OptimizerOptions cannot be instantiated directly. " "Use a subclass such as NFTOptimizerOptions.")
        super().__init__(**data)


class NFTOptimizerOptions(OptimizerOptions):
    """Configuration options for the NFT (Nakanishi-Fujii-Todo) optimizer.

    The NFT algorithm is a gradient-free optimizer designed for variational quantum
    algorithms. For detailed information about the algorithm, see the paper:
    https://arxiv.org/abs/1903.12166

    Preconditions:
        NFT requires the following conditions on the parameterized quantum circuit:

        1. Parameters must be independent: each parameter must appear in exactly one
           gate (no reusing the same parameter across multiple gates).

        2. Parameterized gates must be rotations of the form ``R_j(θ_j) = exp(-i*θ_j*A_j/2)``
           where ``A_j² = I`` (e.g., RX, RY, RZ gates satisfy this).

        3. The cost function must be a sum of expectation values of Hermitian operators:
           ``L(θ) = Σ_k w_k ⟨ψ_k|U†(θ) H_k U(θ)|ψ_k⟩``.

    Scaling:
        NFT updates one parameter at a time. Each full sweep through N parameters
        requires ≥2N function evaluations (depending on reset_interval).

    Args:
        randomized_order: If True, shuffles the order of parameters to update
            each lap (full sweep through all parameters). Default: False.
        reset_interval: How often to reset the recycled loss value.
            Set to 0 to disable resets. Default: 32.
        maxfev: Maximum number of function evaluations (circuit executions).
            Optimization stops when this limit is reached. Default: 1024.
        maxiter: Maximum number of iterations (parameter updates). Default: 500.
        eps: Small epsilon value to avoid division by zero in the analytic
            solution. Default: 1e-32.

    Notes:
        Stopping criterion: Optimization stops when **either** maxfev or maxiter
        is reached, whichever comes first.

        Function evaluations per iteration: Each iteration uses 2-3 function
        evaluations. The very first iteration and the first iteration of each
        reset interval use 3 evaluations. Subsequent iterations reuse the
        previous optimal value, requiring only 2 evaluations.

    Example:
        >>> from haiqu.sdk.qml import NFTOptimizerOptions
        >>> optimizer = NFTOptimizerOptions(maxfev=2048, maxiter=100)
    """

    randomized_order: bool = False
    reset_interval: int = 32
    maxfev: int = 1024
    maxiter: int = 500
    eps: float = 1e-32
