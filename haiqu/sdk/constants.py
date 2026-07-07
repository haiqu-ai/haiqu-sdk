"""
Haiqu SDK. Constants module.
"""

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
