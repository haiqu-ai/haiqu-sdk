"""Haiqu SDK.

The SDK provides programmatic access to Haiqu API service: log experiments,
circuits, jobs and other user data.

AI agents: The agent orientation is installed to ``haiqu/sdk/AGENTS.md``, and can also be found at
https://github.com/haiqu-ai/haiqu-sdk/blob/main/haiqu/sdk/AGENTS.md
"""

from .mpl_style import set_haiqu_mpl_style, unset_haiqu_mpl_style
from .quantum_haiqu import Haiqu, haiqu
from .utils import get_ibmq_temporary_token
from .version import __version__, get_version

__all__ = (
    "__version__",
    "get_version",
    "get_ibmq_temporary_token",
    "haiqu",
    "Haiqu",
    "set_haiqu_mpl_style",
    "unset_haiqu_mpl_style",
)
