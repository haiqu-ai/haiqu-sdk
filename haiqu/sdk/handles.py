"""
Handles are names that users choose for themselves.

Handles identify users in contexts where multiple users might encounter each other.

A handle has 4 to 32 characters: lowercase letters, digits and hyphens. It starts with a letter, and each hyphen sits between two
letters or digits (no repeated, leading, or trailing hyphens), as in ``example-user-2``. A user chooses a handle once
(:meth:`~haiqu.sdk.quantum_haiqu.Haiqu.set_handle`), and cannot change it.
"""

import string
from typing import Annotated

from pydantic import AfterValidator

from ._typecheck import typecheck


HANDLE_MIN_LENGTH = 4
HANDLE_MAX_LENGTH = 32
HANDLE_CHARACTERS = frozenset(string.ascii_lowercase + string.digits + "-")


@typecheck
def check_handle(handle: str) -> str:
    """Return ``handle``, if it is a valid handle.

    Raises:
        ValueError: If ``handle`` is not a valid handle, with the reason.
    """
    if not HANDLE_MIN_LENGTH <= len(handle) <= HANDLE_MAX_LENGTH:
        raise ValueError(f"The handle must have {HANDLE_MIN_LENGTH} to {HANDLE_MAX_LENGTH} characters, not {len(handle)}.")

    if not HANDLE_CHARACTERS.issuperset(handle):
        raise ValueError("The handle must have only lowercase letters, digits and hyphens.")
    elif handle[0] not in string.ascii_lowercase:
        raise ValueError("The handle must start with a lowercase letter.")
    elif handle.endswith("-"):
        raise ValueError("The handle must end with a lowercase letter or a digit.")
    elif "--" in handle:
        raise ValueError("The handle must not have two hyphens in a row.")

    return handle


#: A handle, as a field of a model.
Handle = Annotated[str, AfterValidator(check_handle)]
