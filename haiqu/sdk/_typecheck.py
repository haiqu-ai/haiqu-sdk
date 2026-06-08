"""Shared runtime type-checking decorators for the SDK.

This module keeps beartype configuration local and explicit rather than
enabling package-wide import hooks. That keeps rollout risk low for the SDK:
call sites opt in function by function, and dynamic or legacy surfaces can
remain untouched until they are ready.
"""

from beartype import BeartypeConf, BeartypeStrategy, beartype

# Default decorator for public-facing helpers.
# ``O1`` keeps runtime checks cheap and predictable, which is appropriate for
# SDK utility functions that may sit on common execution paths.
# ``is_pep484_tower=True`` keeps numeric hints pragmatic for users, allowing
# ``int`` values where ``float`` is annotated.
typecheck = beartype(
    conf=BeartypeConf(
        strategy=BeartypeStrategy.O1,
        is_pep484_tower=True,
    )
)
