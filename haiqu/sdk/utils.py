"""
Haiqu SDK: Utilities functions.
"""

import base64
import hashlib
from collections import Counter
from datetime import datetime
import io
import json
import logging
from typing import Any, Union
import os

import openqasm3.parser
import numpy as np
import pandas as pd
import qiskit.qasm2
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import ParameterExpression
from qiskit.primitives import BasePrimitiveJob, EstimatorResult, SamplerResult
from qiskit.primitives.containers import PrimitiveResult
from qiskit.providers import Job
from qiskit.quantum_info import SparsePauliOp
from qiskit.result import Result
from qiskit.result.models import ExperimentResult, ExperimentResultData
from qiskit_qasm3_import import parse as qasm3_parse
from qiskit.transpiler.passes import UnrollCustomDefinitions
from qiskit.transpiler import PassManager
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
from qiskit_aer import AerSimulator

from qiskit_ibm_runtime.api.session import RetrySession
from qiskit_ibm_runtime.api.rest import Api
from qiskit_ibm_runtime.fake_provider import (
    FakeKyiv,  # Eagle
    FakeSherbrooke,
    FakeBrisbane,
    FakeQuebec,
    FakeKyoto,
    FakeWashingtonV2,
    FakeTorino,  # Heron
    FakeFez,
    FakeMarrakesh,
    FakeAlgiers,  # Falcon
    FakeMontrealV2,
    FakeGeneva,
)

from qiskit_algorithms.utils.circuit_key import _circuit_key

MAX_QUBITS = 1024
QPY_DUMP_VERSION = 13

assert (
    qpy.QPY_COMPATIBILITY_VERSION <= QPY_DUMP_VERSION <= qpy.QPY_VERSION
), f"QPY_DUMP_VERSION = {QPY_DUMP_VERSION} is not supported by Qiskit"


class AerSimulatorMoreQubits(AerSimulator):
    """Subclass of the AerSimulator to use 100+ qubits with MPS."""

    def __init__(self, *args, n_qubits=MAX_QUBITS, **kwargs):
        super().__init__(*args, n_qubits=n_qubits, **kwargs)

    def _set_method_config(self, *args, **kwargs):
        # This is called when the method is set (e.g. to "matrix_product_state"), and the implementation in AerSimulator clobbers
        # n_qubits. It doesn't do anything we need, so skip it completely.
        pass


IBM_DEVICES = [
    "ibm_boston",  # Heron
    "ibm_marrakesh",
    "ibm_fez",
    "ibm_torino",
    "ibm_kingston",
    "ibm_pittsburgh",
    "ibm_sherbrooke",  # Eagle
    "ibm_brisbane",
]
IBM_EU_DEVICES = [
    "ibm_aachen",  # Heron
    "ibm_strasbourg",  # Eagle
    "ibm_brussels",
]
IBM_SIMULATORS = {
    "aer_simulator": AerSimulatorMoreQubits,
}
IBM_FAKE_DEVICES = {
    "fake_kyiv": FakeKyiv,
    "fake_sherbrooke": FakeSherbrooke,
    "fake_brisbane": FakeBrisbane,
    "fake_quebec": FakeQuebec,
    "fake_kyoto": FakeKyoto,
    "fake_washington": FakeWashingtonV2,
    "fake_torino": FakeTorino,
    "fake_fez": FakeFez,
    "fake_marrakesh": FakeMarrakesh,
    "fake_algiers": FakeAlgiers,
    "fake_montreal": FakeMontrealV2,
    "fake_geneva": FakeGeneva,
}
AWS_DEVICES = [
    "sv1",
    "lucy",
    "aspen_m3",
    "aria_1",
    "forte_1",
    "garnet",
    "emerald",
    "ankaa_3",
]
AWS_DEFAULT_REGION = "us-east-1"
HAIQU_OPERATIONS = ["HaiquCircuit", "HaiquCircuitdg"]


class HaiquJSONEncoder(json.JSONEncoder):
    """JSON encoder for API client. Handles NumPy arrays."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return to_npy(obj)
        elif isinstance(obj, np.complex128):
            return to_npy(obj)
        return super().default(obj)


def ensure_qc(circuit) -> QuantumCircuit:
    """Make sure circuit is Qiskit QuantumCircuit. Convert if needed."""
    if isinstance(circuit, QuantumCircuit):
        return circuit
    elif isinstance(circuit, str):
        try:
            # Method from_qasm_str uses legacy instructions
            return QuantumCircuit.from_qasm_str(circuit)
            # return qiskit.qasm2.loads(circuit)
        except qiskit.qasm2.exceptions.QASM2ParseError:
            return qasm3_parse(circuit)
    raise TypeError("Circuit must be QASM string or Qiskit QuantumCircuit.")


def from_qpy(encoded: str):
    """Load circuit from base64-encoded qiskit.QPY binary format.

    Args:
        encoded (str): Base64-encoded binary data.

    Returns:
        QuantumCircuit: The decoded circuit.
    """
    buf = io.BytesIO(base64.b64decode(encoded))
    circuits = qpy.load(buf)
    return circuits[0]  # We get a list back even if we only dumped a single circuit.


def to_qpy(circuit: QuantumCircuit):
    """Dump circuit to qiskit.QPY binary format. Then Base64 encode to use in JSON.

    Args:
        circuit (QuantumCircuit): The input circuit, Qiskit object.

    Returns:
        str: Base64 encoded binary data.
    """
    buf = io.BytesIO()
    qpy.dump(circuit, buf, version=QPY_DUMP_VERSION)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def to_npy(arr: np.ndarray):
    """Dump numpy array to NPY binary format. Then Base64 encode to use in JSON.

    Args:
        arr (np.ndarray): The input array, NumPy object.

    Returns:
        str: Base64 encoded binary data.
    """
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def from_npy(encoded: str):
    """Load numpy array from base64-encoded NPY binary format.

    Args:
        encoded (str): Base64-encoded binary data.

    Returns:
        np.ndarray: The decoded array.
    """
    buf = io.BytesIO(base64.b64decode(encoded))
    return np.load(buf)


def to_qasm(circuit):
    """Dump circuit to QASM 2, in case of incompatibility - to QASM 3"""
    try:
        return qiskit.qasm2.dumps(circuit)
    except qiskit.qasm2.QASM2ExportError:
        return qiskit.qasm3.dumps(circuit)


def get_circuit_hash(circuit: QuantumCircuit) -> str:
    """Calculate Hash for qiskit.QuantumCircuit."""
    decompose_gates = HAIQU_OPERATIONS + ["rx", "ry", "rz", "cx"]
    pm = PassManager(
        [UnrollCustomDefinitions(SessionEquivalenceLibrary, decompose_gates)]  # will only unroll ops that have definitions
    )
    circuit = pm.run(circuit)

    if circuit.num_parameters > 0:
        circuit = circuit.assign_parameters([0] * circuit.num_parameters)

    circuit_hash = _circuit_key(circuit)
    hash_object = hashlib.sha256(repr(circuit_hash).encode("utf-8"))
    hash_bytes = hash_object.digest()
    hash_value = int.from_bytes(hash_bytes, byteorder="little")

    return str(hash_value)


def find_shared_parameters(circuit: QuantumCircuit) -> list[str]:
    """Find parameters that are used in more than one gate.

    Args:
        circuit: A parameterized QuantumCircuit.

    Returns:
        List of parameter names that appear in multiple gates.
    """
    param_counts: Counter = Counter()

    for inst in circuit.data:
        for param in inst.operation.params:
            if isinstance(param, ParameterExpression):
                for p in param.parameters:
                    param_counts[p.name] += 1

    return [name for name, count in param_counts.items() if count > 1]


def get_job_hash(job_data) -> str:
    """Calculate Hash for the Qiskit results object."""
    hash_object = hashlib.sha256(b"")
    values = []

    if isinstance(job_data, Result):
        values.append(job_data.job_id)
    else:
        values.append(repr(job_data))

    for value in values:
        encoded_value = repr(value).encode("utf-8")
        hash_object.update(encoded_value)

    hash_bytes = hash_object.digest()
    hash_value = int.from_bytes(hash_bytes, byteorder="little")
    return str(hash_value)


def get_job_results(job_data):
    """Extract actual results dataset from the local job/results/sampler/etc."""
    if isinstance(job_data, Result):
        return job_data.get_counts()
    elif isinstance(job_data, (ExperimentResult, ExperimentResultData)):
        return job_data.to_dict()
    elif isinstance(job_data, (Job, BasePrimitiveJob)):
        res = job_data.result()
        if isinstance(res, (Result, ExperimentResult, ExperimentResultData)):
            return res.to_dict()
        return res
    elif isinstance(job_data, EstimatorResult):
        return job_data.values.tolist()  # np array
    elif isinstance(job_data, SamplerResult):
        return job_data.quasi_dists  # list of dicts
    elif isinstance(job_data, PrimitiveResult):
        return [x for x in job_data]
    return getattr(job_data, "data", "")


def get_job_shots(job_data):
    """Extract shots number from the local job/results/sampler/etc."""
    if isinstance(job_data, Result):
        return [x.shots for x in job_data.results]
    elif isinstance(job_data, SamplerResult):
        return [x["shots"] for x in job_data.metadata]
    return getattr(job_data, "shots", "N/A")


def get_job_device(job_data):
    """Extract device information from the local job/results/sampler/etc."""
    if isinstance(job_data, Job):
        backend = job_data.backend()
        if backend.version == 1:
            return backend.name()
        return backend.name
    elif isinstance(job_data, Result):
        return job_data.backend_name
    elif isinstance(job_data, SamplerResult):
        return repr([x["simulator_metadata"]["device"] for x in job_data.metadata])
    return "N/A"


def get_job_name(job_data):
    """Extract bits to name this job in a human-friendly way."""
    if isinstance(job_data, Job):
        return f"Local job {job_data.job_id()}"
    elif isinstance(job_data, Result):
        return f"Local job {job_data.job_id} - Job result, histogram data of an experiment"
    elif isinstance(job_data, SamplerResult):
        return "Local job - Sampler result, quasi-probabilities"
    return "Local job"


def get_ibmq_temporary_token(user_token: str, ibmq_api_url: str = "https://auth.quantum-computing.ibm.com/api") -> dict:
    """
    Login to IBMQ account using User Token and generate a Temporary Token.

    Args:
        user_token (str): User's IBMQ API Token from https://quantum.ibm.com/
        ibmq_api_url (str): IBMQ API URL.

    Returns:
        A response dictionary with Temporary Token, it's Creation Time
        and Time-To-Live.
    """

    ibmq_api = Api(RetrySession(base_url=ibmq_api_url))

    response = ibmq_api.login(user_token)
    response["token"] = response.pop("id")

    return response


def sparse_op_to_tuple(sparse_op):
    """
    Convert a single SparsePauliOp to a tuple of (paulis, coeffs).

    Args:
        sparse_op: A single SparsePauliOp object

    Returns:
        tuple: (paulis, coeffs) where paulis is list of Pauli strings and coeffs is list of real coefficients
    """
    if not isinstance(sparse_op, SparsePauliOp):
        raise ValueError("sparse_op must be a single SparsePauliOp object")

    paulis = []
    coeffs = []

    for term, coeff in sparse_op.to_list():
        paulis.append(term)
        assert np.isclose(coeff.imag, 0.0), (
            "Observable must be a valid Hermitian operator with real coefficients. "
            f"Found {term} with complex coefficient {coeff}"
        )
        coeffs.append(coeff.real)

    return (paulis, coeffs)


def validate_and_normalize_parameters_and_observables(parameters, observables, num_circuits):
    """Validate and normalize ``parameters`` and ``observables`` for circuit submission.

    Coerces user-provided shapes into the canonical internal forms:

    - ``parameters``: 3D list of floats ``[[[..], [..]], ...]`` (one parameter group per circuit).
    - ``observables``: 3D list of ``(paulis, coeffs)`` tuples ``[[tup, ...], ...]``
      (one inner list per circuit).

    Accepted ``observables`` input shapes:

    - **Single circuit** (``num_circuits == 1``):
        - ``SparsePauliOp`` — one observable on the circuit.
        - ``[[op1, op2, ...]]`` — nested form, one or more observables on the circuit.
        - ``[op1, op2, ...]`` — bare list of observables.
    - **Multiple circuits** (``num_circuits > 1``):
        A list of length ``num_circuits`` where each element is independently either a bare
        ``SparsePauliOp`` (one observable on that circuit) or a ``list[SparsePauliOp]`` (multiple
        observables on that circuit). Mixing is allowed — e.g., ``[[op1, op2], op3]`` for two circuits
        means circuit 0 has two observables and circuit 1 has one.

    The fully-nested form ``[[..], [..], ...]`` is the unambiguous canonical shape and is preferred for
    callers that handle both single and multi-circuit cases through the same code path.

    Args:
        parameters: 2D float list (single circuit) or 3D float list (multiple circuits), or ``None``.
        observables: One of the accepted shapes above, or ``None``.
        num_circuits: Number of circuits being submitted.

    Returns:
        Tuple ``(parameters, observables)`` in canonical internal form.

    Raises:
        ValueError: If the input shape doesn't match any accepted form.
    """

    def is_float_list(lst):
        return isinstance(lst, list) and all(isinstance(x, (int, float)) for x in lst)

    def is_sparse_list(lst):
        return isinstance(lst, list) and all(isinstance(o, SparsePauliOp) for o in lst)

    # Parameters validation - accepts either 2D or 3D
    if parameters is not None:
        if not isinstance(parameters, list):
            raise ValueError("`parameters` must be a list.")

        if num_circuits == 1:
            # Single circuit: accept 2D [[p1, p2], [q1, q2], ...]
            if not all(is_float_list(row) for row in parameters):
                raise ValueError("Single circuit: `parameters` must be 2D list of floats, " "[[p1, p2], [q1, q2], ...].")
        else:
            # Multiple circuits: accept 3D [[[c1p1, c1p2], [c1q1, c1q2]], [[c2r1, c2r2], [c2s1, c2s2]]]
            if len(parameters) != num_circuits:
                raise ValueError(f"Multiple circuits: `parameters` must have {num_circuits} entries, one per circuit.")

            if not all(all(is_float_list(row) for row in group) for group in parameters):
                raise ValueError(
                    "Multiple circuits: `parameters` must be 3D list of floats, "
                    "[[[c1p1, c1p2], [c1q1, c1q2]], [[c2r1, c2r2], [c2s1, c2s2]]]."
                )

    # Observables validation - always return [[tuples]] format
    if observables is not None:
        if num_circuits == 1:
            # Single circuit - always return [[tuple1, tuple2, ...]]
            if isinstance(observables, SparsePauliOp):
                # Single operator → [[tuple]]
                observables = [[sparse_op_to_tuple(observables)]]
            elif isinstance(observables, list) and len(observables) == 1 and is_sparse_list(observables[0]):
                # Nested form [[op1, op2, ...]] → [[tuple1, tuple2, ...]]
                observables = [[sparse_op_to_tuple(op) for op in observables[0]]]
            elif is_sparse_list(observables):
                # Bare list of operators → [[tuple1, tuple2, ...]]
                observables = [[sparse_op_to_tuple(op) for op in observables]]
            else:
                raise ValueError(
                    "Single circuit: `observables` must be a SparsePauliOp, a nested list "
                    "[[op1, op2, ...]], or a bare list of SparsePauliOp objects."
                )
        else:
            # Multiple circuits - return [[tuples_for_c1], [tuples_for_c2], ...]
            if not isinstance(observables, list) or len(observables) != num_circuits:
                raise ValueError(f"Multiple circuits: `observables` must have {num_circuits} entries, one per circuit.")

            processed = []
            for circuit_idx, circuit_obs in enumerate(observables):
                if isinstance(circuit_obs, SparsePauliOp):
                    # Single operator for this circuit → [tuple]
                    processed.append([sparse_op_to_tuple(circuit_obs)])
                elif is_sparse_list(circuit_obs):
                    # List of operators for this circuit → [tuple1, tuple2, ...]
                    processed.append([sparse_op_to_tuple(op) for op in circuit_obs])
                else:
                    raise ValueError(
                        f"Circuit {circuit_idx}: Expected SparsePauliOp or list of SparsePauliOp objects, "
                        f"but received {type(circuit_obs).__name__}: {circuit_obs}"
                    )
            observables = processed

    return parameters, observables


def is_haiqu_generated(circuit: QuantumCircuit) -> bool:
    """
    Check if the circuit contains HaiquCircuitGate, indicating it was generated by Haiqu.

    Args:
        circuit (QuantumCircuit): The input circuit.
    Returns:
        bool: True if the circuit contains HaiquCircuitGate, False otherwise.
    """
    for instruction in circuit.data:
        if instruction.operation.name in HAIQU_OPERATIONS:
            return True
    return False


def preprocess_metrics(**kwargs) -> dict:
    """
    Preprocess user-defined metrics dictionary.
    These metrics are arbitrary key-value pairs provided by the user.
    If any value is a Drawer object or a Matplotlib plot object, it is serialized
    to a base64-encoded PNG string.

    Returns:
        dict: The metrics dictionary.
    """
    from haiqu.sdk.wiz.drawer import Drawer

    processed_metrics = {}
    prefix = "data:image/png;base64,"

    for key, value in kwargs.items():
        if isinstance(value, Drawer):  # Drawer image object
            processed_metrics[key] = prefix + base64.b64encode(value.fig.to_image(format="png")).decode("utf-8")
        elif isinstance(value, pd.DataFrame):  # Pandas DataFrame
            processed_metrics[key] = value.to_dict()
        elif hasattr(value, "savefig"):  # Matplotlib plot object
            buf = io.BytesIO()
            value.savefig(buf, format="png")
            buf.seek(0)
            processed_metrics[key] = prefix + base64.b64encode(buf.read()).decode("utf-8")
        else:
            processed_metrics[key] = value

    return processed_metrics


def generate_artifact_name(parent_ctx: Any, child_ctx: Any) -> str:
    """Generate an artifact name based on user input."""
    from haiqu.sdk.wiz.drawer import Drawer
    from haiqu.sdk.wiz.jupyter import DATE_TIME_FORMAT

    prefix = "Artifact"

    if isinstance(parent_ctx, Drawer) or isinstance(child_ctx, Drawer):  # Drawer image object
        prefix = "Plot"
    elif hasattr(parent_ctx, "savefig") or hasattr(child_ctx, "savefig"):  # Matplotlib plot object
        prefix = "Plot"
    elif isinstance(parent_ctx, pd.DataFrame) or isinstance(child_ctx, pd.DataFrame):
        prefix = "DataFrame"

    return f"{prefix} {datetime.now().strftime(DATE_TIME_FORMAT)}"


def check_circuit_context(ctx: Any) -> Union[QuantumCircuit, None]:
    """Check if the context is a QuantumCircuit or QASM string."""
    if isinstance(ctx, str):
        # Test if string is a QASM 2.0 / 3.0 dump
        try:
            return ensure_qc(ctx)
        except openqasm3.parser.QASM3ParsingError:
            return None
    elif isinstance(ctx, QuantumCircuit):
        return ctx

    return None


def is_jupyter() -> bool:
    """Return True when running inside a Jupyter Lab/Notebook environment."""
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        return ipy is not None and "text/html" in ipy.display_formatter.active_types
    except Exception:
        return False


def detect_notebook() -> str | None:
    """Detect if we are running inside a Jupyter Notebook environment.

    Returns the notebook path if detected, otherwise None.
    Example return value: /examples/run_examples/200_GHZStatePreparation.ipynb
    """
    if not is_jupyter():
        return

    full_path = globals().get("__session__") or os.environ.get("JPY_SESSION_NAME")
    # Example: /user_storage/examples/run_examples/200_GHZStatePreparation.ipynb

    if full_path is None:
        return

    bits = full_path.split("user_storage")
    if len(bits) != 2:
        # Unable to parse notebook path, return full path as is
        # Let Dashboard handle it (useful for local runs)
        return full_path

    return bits[1]


def setup_logger(logger_name: str, log_level=logging.INFO) -> logging.Logger:
    """Initialize the logger."""
    LOG_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
