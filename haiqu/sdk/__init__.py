"""Haiqu SDK.

The SDK provides programmatic access to Haiqu API service: log experiments,
circuits, jobs and other user data.

AI agents: The agent orientation is installed to ``haiqu/sdk/AGENTS.md``, and can also be found at
https://github.com/haiqu-ai/haiqu-sdk/blob/main/haiqu/sdk/AGENTS.md
"""

from .constants import Verbosity
from .mpl_style import set_haiqu_mpl_style, unset_haiqu_mpl_style
from .quantum_haiqu import Haiqu, haiqu
from .schemas import (
    AnalyticalSynthesisOptions,
    BlockwiseSynthesisOptions,
    DataLoadingSynthesisOptions,
    VariationalSynthesisOptions,
)
from .version import __version__, get_version

__all__ = (
    "__version__",
    "get_version",
    "haiqu",
    "Haiqu",
    "Verbosity",
    "AnalyticalSynthesisOptions",
    "BlockwiseSynthesisOptions",
    "DataLoadingSynthesisOptions",
    "VariationalSynthesisOptions",
    "set_haiqu_mpl_style",
    "unset_haiqu_mpl_style",
)
