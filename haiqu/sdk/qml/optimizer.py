"""
Haiqu SDK QML: Optimizer configuration classes.
"""

from __future__ import annotations

from typing import Annotated, Any, Dict, Literal, Union

from pydantic import BaseModel, Field, model_validator


class OptimizerOptions(BaseModel):
    """Base class for optimizer configuration.

    Subclass this to create configuration for specific optimizers.
    Do not instantiate directly; use a subclass like NFTOptimizerOptions or
    ScipyOptimizerOptions.
    """

    def __init__(self, **data):
        if type(self) is OptimizerOptions:
            raise TypeError(
                "OptimizerOptions cannot be instantiated directly. "
                "Use a subclass such as NFTOptimizerOptions or ScipyOptimizerOptions."
            )
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
            Optimization stops when this limit is reached. Default: 200.
        maxiter: Maximum number of iterations (parameter updates). Default: 100.
        eps: Small epsilon value to avoid division by zero in the analytic
            solution. Default: 1e-32.

    Notes:
        Stopping criterion: Optimization stops when **either** maxfev or maxiter
        is reached, whichever comes first.

        Function evaluations per iteration: Each iteration uses 2-3 function
        evaluations. The very first iteration and the first iteration of each
        reset interval use 3 evaluations. Subsequent iterations reuse the
        previous optimal value, requiring only 2 evaluations.

        Result histories: ``job.result().loss_history`` and
        ``weights_history`` record one entry per NFT iteration (parameter
        update), not per circuit evaluation. History length therefore tracks
        ``maxiter``, not ``maxfev``.

    Example:
        >>> from haiqu.sdk.qml import NFTOptimizerOptions
        >>> optimizer = NFTOptimizerOptions(maxfev=500, maxiter=200)
    """

    type: Literal["nft"] = "nft"
    randomized_order: bool = False
    reset_interval: int = 32
    maxfev: int = 200
    maxiter: int = 100
    eps: float = 1e-32


# Per-method whitelist of keys that may appear in ScipyOptimizerOptions.options.
# 'maxfev' is excluded because it is exposed as a typed top-level field. Update
# this dict when adding a new supported method.
_KNOWN_OPTIONS = {
    "cobyla": {"rhobeg", "tol", "maxiter", "catol", "disp", "f_target"},
    "nelder-mead": {"xatol", "fatol", "adaptive", "maxiter", "disp"},
    "powell": {"xtol", "ftol", "direc", "maxiter", "disp"},
    "cobyqa": {"rhobeg", "final_tr_radius", "maxiter", "f_target", "disp"},
}


class ScipyOptimizerOptions(OptimizerOptions):
    """Configuration for any derivative-free ``scipy.optimize.minimize`` method.

    The Haiqu backend wraps ``scipy.optimize.minimize`` for the four supported
    derivative-free methods. ``maxfev`` is the only option that is universal
    across all of them, so it gets a typed slot; everything else goes in the
    free-form ``options`` dict and is validated against a per-method whitelist
    at construction time, mirroring what ``scipy.optimize.minimize`` accepts.

    Methods:
        - ``cobyla``: Constrained Optimization BY Linear Approximation. Trust-region
          method with linear surrogates; robust default on noisy expectation values.
        - ``nelder-mead``: Downhill simplex. No surrogate model; forgiving on noisy
          or non-smooth objectives but tends to need more evaluations.
        - ``powell``: Direction-set method that minimizes along conjugate directions;
          often fast on well-conditioned problems.
        - ``cobyqa``: COBYLA's quadratic-approximation successor; typically higher
          quality per evaluation than COBYLA at modest extra cost.

    Args:
        method: scipy method name. One of ``cobyla``, ``nelder-mead``, ``powell``, ``cobyqa``.
        maxfev: Maximum number of function evaluations (circuit executions).
            Default: 200. The Haiqu backend enforces this cap uniformly across
            methods even when scipy's native option name differs.
        options: Per-method options forwarded to ``scipy.optimize.minimize``.
            Allowed keys are validated at construction time; an unknown key raises
            ``ValueError``.

            Per-method allowed keys:

            - ``cobyla``: ``rhobeg``, ``tol``, ``maxiter``, ``catol``, ``disp``,
              ``f_target``
            - ``nelder-mead``: ``xatol``, ``fatol``, ``adaptive``, ``maxiter``,
              ``disp``
            - ``powell``: ``xtol``, ``ftol``, ``direc``, ``maxiter``, ``disp``
            - ``cobyqa``: ``rhobeg``, ``final_tr_radius``, ``maxiter``,
              ``f_target``, ``disp``

            ``maxfev`` is intentionally excluded; pass it via the top-level field.
            ``maxiter`` is allowed for every method so scipy's own iteration /
            evaluation budget can be raised when it would otherwise stop before
            Haiqu's ``maxfev``.

    Notes:
        Final result selection: scipy methods can wander after they have found a
        good point. Haiqu therefore returns the best-so-far parameters tracked
        across the optimization, not the final scipy iterate.

        Result histories: ``job.result().loss_history`` and
        ``weights_history`` record one entry per objective / circuit
        evaluation, not per scipy iteration. Length is capped by ``maxfev`` and
        can exceed ``options["maxiter"]`` (e.g. Nelder–Mead / COBYQA init
        batches dump n+1 or 2n+1 entries on the first tick).

    Example:
        >>> from haiqu.sdk.qml import ScipyOptimizerOptions
        >>> ScipyOptimizerOptions(method="cobyla", maxfev=200, options={"rhobeg": 0.5})
        >>> ScipyOptimizerOptions(
        ...     method="cobyla",
        ...     maxfev=2000,
        ...     options={"rhobeg": 0.3, "tol": 1e-8, "maxiter": 2000},
        ... )
        >>> ScipyOptimizerOptions(method="powell", maxfev=500, options={"xtol": 1e-6})
        >>> ScipyOptimizerOptions(method="nelder-mead", options={"adaptive": True})
    """

    type: Literal["scipy"] = "scipy"
    method: Literal["cobyla", "nelder-mead", "powell", "cobyqa"]
    maxfev: int = 200
    options: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_options(self) -> "ScipyOptimizerOptions":
        allowed = _KNOWN_OPTIONS[self.method]
        unknown = set(self.options) - allowed
        if unknown:
            raise ValueError(
                f"Unknown options for method={self.method!r}: {sorted(unknown)}. "
                f"Allowed: {sorted(allowed)}. (maxfev is a top-level field, not an option key.)"
            )
        return self


OptimizerOptionsUnion = Annotated[
    Union[NFTOptimizerOptions, ScipyOptimizerOptions],
    Field(discriminator="type"),
]
