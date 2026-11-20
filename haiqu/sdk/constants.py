"""
Haiqu SDK. Constants module.
"""

import enum

REST_API_URI = "https://api.haiqu.ai/"

SDK_VERSION_HEADER = "haiqu-sdk-version"

# The URI of the Haiqu Dashboard. This is used to generate links to experiments, jobs, and circuits.
DASHBOARD_URI = "https://dashboard.haiqu.ai"
DASHBOARD_LOCAL_PORT = 8080

DASHBOARD_EXPERIMENT_PATH = "/experiment/{experiment_id}"
DASHBOARD_JOB_PATH = "/jobs/{job_id}"
DASHBOARD_CIRCUIT_PATH = "/circuits/{circuit_id}"

# Maximum soft time limit (seconds) accepted for data-loading jobs.
MAX_DATA_LOADING_TIME = 900  # 15 min
# Maximum soft time limit (seconds) accepted for state-compression jobs.
MAX_COMPRESSION_TIME = 3600  # 1 hour

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
# Maximum per-layer bond dimension for circuit synthesis. The disentangling gates are
# ceil(log2(bond_dim)) + 1 qubits wide, so this caps them at 5 qubits.
MAX_DL_BOND_DIM_LAYER = 16
# Smallest accepted width (interval_end - interval_start) for interval-based loaders
# (distribution and function loading).
MIN_DL_INTERVAL_WIDTH = 1e-8

# Range of the state-compression approximation level.
MIN_COMPRESSION_APPROXIMATION_LEVEL = 1
MAX_COMPRESSION_APPROXIMATION_LEVEL = 100

# Maximum length (characters) of a user-supplied math expression string, e.g. a
# variational loss expression or a function-loading formula.
MAX_MATH_EXPRESSION_LENGTH = 2048

# Maximum length of an observable term string (one character per qubit).
MAX_OBSERVABLE_QUBITS = 1000


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


# Named basis families a function-tomography fit can be expressed in.
TOMOGRAPHY_FAMILIES = ("legendre", "chebyshev_t", "fourier")

# Maximum qubits encoding one spatial axis of a tomography register.
MAX_TOMOGRAPHY_AXIS_QUBITS = 40

# Maximum register points a custom tomography grid may hold; every point's coordinates travel in the request.
MAX_TOMOGRAPHY_GRID_POINTS = 2**20


class Verbosity(enum.IntEnum):
    """How much the SDK displays besides the values its methods return."""

    LOW = 0
    NORMAL = 1
    HIGH = 2
