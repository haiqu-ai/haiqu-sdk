"""
Haiqu SDK. Constants module.
"""

import enum

REST_API_URI = "https://api.haiqu.ai/"

SDK_VERSION_HEADER = "haiqu-sdk-version"


DASHBOARD_EXPERIMENT_SCHEMA = "https://dashboard.haiqu.ai/experiment/{experiment_id}"

# Maximum soft time limit (seconds) accepted for data-loading jobs.
MAX_DATA_LOADING_TIME = 900  # 15 min
# Maximum soft time limit (seconds) accepted for state-compression jobs.
MAX_COMPRESSION_TIME = 1200  # 20 min

# Maximum size (bytes) of notebook/script source logged.
MAX_SOURCE_FILE_SIZE = 5_000_000  # 5 MB

# The fit builds a dense 2^n x 2^n unitary; cap n on server memory (not an API limit).
MAX_SU2_EQUIVARIANT_COMPILATION_QUBITS = 10

MAX_ORBITALS = 256  # h2e scales as norb**4, so large values exhaust memory
MAX_COMPRESSION_QUBITS = 1000  # cap for compression qubit counts
MAX_DL_QUBITS = 1000  # cap for data-loading qubit counts

# Number of variables supported by multivariate distribution loading.
SUPPORTED_MULTIVARIATE_DIMENSIONS = 2

# Maximum number of qubits in a vector-loading circuit (amplitude encoding).
MAX_DL_VECTOR_QUBITS = 20
# Maximum length of the vector accepted by vector loading (one amplitude per basis state).
MAX_DL_VECTOR_LENGTH = 2**MAX_DL_VECTOR_QUBITS
# Maximum number of layers in any data-loading circuit.
MAX_DL_LAYERS = 100
# Maximum number of post-layer fine-tuning iterations for data loading (0 disables fine-tuning).
MAX_DL_FINE_TUNING_ITERATIONS = 500
# Maximum feature density for entangled-manifold embedding.
MAX_DL_EME_DENSITY = 8
# Maximum bond dimension accepted by MPS loading.
MAX_DL_MPS_BOND_DIMENSION = 64
# Smallest accepted width (interval_end - interval_start) for interval-based loaders
# (distribution and function loading).
MIN_DL_INTERVAL_WIDTH = 1e-8

# Range of the state-compression approximation level.
MIN_COMPRESSION_APPROXIMATION_LEVEL = 1
MAX_COMPRESSION_APPROXIMATION_LEVEL = 100


class CompressionLevel(str, enum.Enum):
    """Qualitative amount of a circuit that state compression targets."""

    LOW = "low"
    BALANCED = "balanced"
    HIGH = "high"
    MAX = "max"


class CompressionFineTuning(str, enum.Enum):
    """How much fine-tuning state compression applies after the initial result."""

    DISABLED = "disabled"
    LOW = "low"
    HEAVY = "heavy"
