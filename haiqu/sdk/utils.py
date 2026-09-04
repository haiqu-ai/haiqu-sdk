"""
Haiqu SDK: Utilities functions.
"""

import base64
import hashlib
from collections import Counter
from collections.abc import Sequence
from datetime import datetime
import functools
import io
import json
import logging
import os
import sys
import warnings
from typing import Any
from urllib.parse import urljoin

import openqasm3.parser
import numpy as np
import pandas as pd
import requests
import qiskit.qasm2
from ._typecheck import typecheck
from .constants import MAX_SOURCE_FILE_SIZE
from qiskit import QuantumCircuit, qpy
import qiskit.circuit
from qiskit.circuit import Instruction, ParameterExpression
import qiskit.circuit.controlflow
import qiskit.circuit.library
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
    "aria_1",
    "forte_1",
    "garnet",
    "emerald",
    "cepheus_1",
]
AWS_DEFAULT_REGION = "us-east-1"
IQM_DEFAULT_SERVER_URL = "https://resonance.iqm.tech"
HAIQU_OPERATIONS = ["HaiquCircuit", "HaiquCircuitdg"]


class HaiquJSONEncoder(json.JSONEncoder):
    """JSON encoder for API client. Handles NumPy arrays."""

    def default(self, obj: Any) -> Any:
        """Serialize supported NumPy values into JSON-compatible payloads.

        This encoder serializes ``np.ndarray`` instances and ``np.complex128``
        scalars as base64-encoded NPY strings so they can be embedded safely in
        JSON request bodies.

        Args:
            obj (Any): Object that the JSON encoder is attempting to serialize.

        Returns:
            Any: Base64-encoded NPY string for supported NumPy values, or the
            result of ``json.JSONEncoder.default`` for other objects.

        Raises:
            TypeError: If ``obj`` is not handled by this method and the parent
            JSON encoder cannot serialize it.
        """
        if isinstance(obj, np.ndarray):
            return to_npy(obj)
        if isinstance(obj, np.complex128):  # JSON serialization can surface NumPy complex scalars element-wise.
            return to_npy(np.asarray(obj))
        return super().default(obj)


@typecheck
def ensure_qc(circuit: QuantumCircuit | str) -> QuantumCircuit:
    """Return a Qiskit circuit from a circuit object or QASM source.

    String inputs are parsed as QASM 2 first and then retried as QASM 3 if the
    QASM 2 parser rejects them.

    Args:
        circuit (QuantumCircuit | str): Existing ``QuantumCircuit`` instance or
            a QASM 2/QASM 3 string.

    Returns:
        QuantumCircuit: The original circuit or a parsed ``QuantumCircuit`` instance.

    Raises:
        openqasm3.parser.QASM3ParsingError: If a string input is not valid QASM.
    """
    if isinstance(circuit, QuantumCircuit):
        return circuit

    try:
        # Method from_qasm_str uses legacy instructions
        return QuantumCircuit.from_qasm_str(circuit)
        # return qiskit.qasm2.loads(circuit)
    except qiskit.qasm2.exceptions.QASM2ParseError:
        return qasm3_parse(circuit)


@typecheck
def from_qpy(encoded: str) -> QuantumCircuit:
    """Decode a circuit from base64-encoded QPY data.

    Args:
        encoded (str): Base64-encoded QPY payload.

    Returns:
        QuantumCircuit: The decoded ``QuantumCircuit``.
    """
    buf = io.BytesIO(base64.b64decode(encoded))
    circuits = qpy.load(buf)
    return circuits[0]  # We get a list back even if we only dumped a single circuit.


def _qpy_check_name_collision(operation):
    """Determine whether ``operation`` will confuse ``_write_instruction`` in ``qiskit/qpy/binary_io/circuits.py`` by having the
    same class name as a standard class.

    This is only a sketch of what the actual implementation checks, so it may need to be extended to handle more interesting
    cases.
    """
    if not isinstance(operation, Instruction):
        return False  # No handling of non-Instruction Operation subclasses

    gate_class = operation.base_class

    for module in [qiskit.circuit.library, qiskit.circuit, qiskit.circuit.controlflow]:
        if hasattr(module, gate_class.__name__) and getattr(module, gate_class.__name__) is not gate_class:
            return True  # The name is the same, but the class is different

    return False


@functools.cache
def _qpy_generate_subclass(cls):
    """Produce a subclass of ``cls`` whose name shouldn't collide with anything in Qiskit."""
    return type(
        cls.__name__ + "__haiqu_qpy",
        (cls,),
        {"base_class": property(lambda self: type(self))},
    )


@typecheck
def to_qpy(circuit: QuantumCircuit, *, check_load: bool = True) -> str:
    """Encode a circuit as base64-wrapped QPY data.

    Args:
        circuit (QuantumCircuit): Circuit to serialize.
        check_load (bool): Whether to verify that the QPY-serialized circuit can be loaded. Defaults to ``True``.

    Returns:
        str: Base64-encoded QPY payload suitable for JSON transport.
    """
    has_collision = False
    for instruction in circuit.data:
        if _qpy_check_name_collision(instruction.operation):
            has_collision = True
            break

    if has_collision:
        out = circuit.copy_empty_like()

        for instruction in circuit.data:
            if _qpy_check_name_collision(instruction.operation):
                # The QPY serializer assumes that any operation whose class name it recognizes must be an instance of the standard
                # operation class, and it doesn't bother serializing the definition. Catch instances of this (e.g. for MSGate in
                # qiskit-ionq) and trick the serializer into treating it as a custom gate by modifying the class name.
                operation = instruction.operation.to_mutable()
                operation.__class__ = _qpy_generate_subclass(operation.__class__)
                instruction = instruction.replace(operation=operation)

                # TODO: Recurse into operations that can contain gates (e.g. custom definition, AnnotatedOperation, ControlFlowOp)

            out.append(instruction)

        for instruction in out.data:
            if _qpy_check_name_collision(instruction.operation):
                raise RuntimeError(f"Failed to resolve name collision for operation: {instruction.operation.base_class.__name__}")
    else:
        out = circuit

    buf = io.BytesIO()
    qpy.dump(out, buf, version=QPY_DUMP_VERSION)

    if check_load:
        buf.seek(0)
        qpy.load(buf)

    return base64.b64encode(buf.getvalue()).decode("utf-8")


@typecheck
def to_npy(arr: np.ndarray) -> str:
    """Encode a NumPy array as base64-wrapped NPY data.

    Args:
        arr (np.ndarray): Array to serialize.

    Returns:
        str: Base64-encoded NPY payload suitable for JSON transport.
    """
    buf = io.BytesIO()
    np.save(buf, arr)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


@typecheck
def from_npy(encoded: str) -> np.ndarray:
    """Decode a NumPy array from base64-encoded NPY data.

    Args:
        encoded (str): Base64-encoded NPY payload.

    Returns:
        np.ndarray: The decoded NumPy array.
    """
    buf = io.BytesIO(base64.b64decode(encoded))
    return np.load(buf)


@typecheck
def compute_fourier_coefficients(
    state: np.ndarray | Sequence,
    freq_ranges: Sequence[Sequence[int]],
    return_fidelity: bool = False,
) -> tuple[np.ndarray, list[int]] | tuple[np.ndarray, list[int], float]:
    """Compute truncated, normalized Fourier coefficients of an n-dimensional array.

    The returned values are the input that
    :meth:`haiqu.sdk.quantum_haiqu.Haiqu.fourier_loading` expects: the coefficients ascend in
    frequency along every axis, which is not the order ``np.fft.fftn`` returns, and ``min_freqs``
    records which frequency the first coefficient of each axis carries.

    Keeping only a few frequencies per axis is what makes Fourier loading cheap, and it is also
    the only lossy step performed here. The optional fidelity quantifies exactly that loss. It is
    a purely classical quantity computed locally and for free, and it is *not* part of the loading
    job's ``quality``, which measures how well the cloud reproduces whatever coefficients it is
    given. The fidelity of the final state against the original ``state`` is approximately the
    product of the two.

    Args:
        state (np.ndarray | Sequence): Values of the target on a (possibly coarse) uniform grid.
            The array is treated as periodic on that grid, so a non-periodic target may need many
            frequencies; symmetrizing it along a dimension can help.
        freq_ranges (Sequence[Sequence[int]]): One ``(min_freq, max_freq)`` pair per dimension,
            giving the inclusive frequency band to retain. Each band must lie inside the range the
            grid resolves, ``[-(n // 2), n - n // 2 - 1]`` for an axis of length ``n``.
        return_fidelity (bool): Whether to also return the retained fraction of the spectral
            energy. Defaults to ``False``.

    Returns:
        tuple[np.ndarray, list[int]] | tuple[np.ndarray, list[int], float]: The complex
        coefficient array of shape ``max_freq - min_freq + 1`` per dimension, normalized to unit
        norm, and the per-dimension ``min_freq`` values. With ``return_fidelity`` a third value is
        appended: the squared overlap between the reconstruction from the retained frequencies and
        ``state``. It is ``1.0`` when the bands cover the whole resolved spectrum, and also when
        they cover the entire support of a finite Fourier series, however few frequencies that is.

    Raises:
        ValueError: If the number of bands does not match the number of dimensions, a band is not
            a pair, ``min_freq`` exceeds ``max_freq``, or a band falls outside the resolved range.

    Examples:
        >>> import numpy as np
        >>> from haiqu.sdk.utils import compute_fourier_coefficients
        >>> grid = np.arange(64) / 64
        >>> cosine = np.cos(2 * np.pi * 3 * grid)  # only the -3 and +3 harmonics are non-zero
        >>> coefficients, min_freqs, fidelity = compute_fourier_coefficients(
        ...     cosine, freq_ranges=[(-3, 3)], return_fidelity=True
        ... )
        >>> print(coefficients.shape, min_freqs, f"{fidelity:.6f}")
        (7,) [-3] 1.000000
    """
    values = np.asarray(state)
    sizes = values.shape
    if len(freq_ranges) != len(sizes):
        raise ValueError(f"freq_ranges must have one (min_freq, max_freq) pair per dimension, got {len(freq_ranges)}")

    # Frequencies index into the spectrum, so they are validated here rather than relying on the
    # type hint: beartype's O1 strategy samples one entry of a container instead of checking all.
    bands = []
    for dimension, band in enumerate(freq_ranges):
        if len(band) != 2:
            raise ValueError(f"freq_ranges[{dimension}] must be a (min_freq, max_freq) pair, got {len(band)} values")
        if not all(isinstance(freq, (int, np.integer)) and not isinstance(freq, (bool, np.bool_)) for freq in band):
            raise ValueError(f"freq_ranges[{dimension}] must hold two integer frequencies, got {tuple(band)}")
        min_freq, max_freq = int(band[0]), int(band[1])
        if min_freq > max_freq:
            raise ValueError(f"Dimension {dimension}: min_freq={min_freq} exceeds max_freq={max_freq}")
        size = sizes[dimension]
        lowest, highest = -(size // 2), size - size // 2 - 1
        if min_freq < lowest or max_freq > highest:
            raise ValueError(
                f"Dimension {dimension} has size {size}, so the frequency range [{min_freq}, {max_freq}] is "
                f"outside the range it resolves, [{lowest}, {highest}]"
            )
        bands.append((min_freq, max_freq))

    spectrum = np.fft.fftshift(np.fft.fftn(values))
    retained = spectrum[tuple(slice(sizes[d] // 2 + low, sizes[d] // 2 + high + 1) for d, (low, high) in enumerate(bands))]

    norm = np.linalg.norm(retained)
    if norm <= 1e-8:
        raise ValueError("The retained frequency bands hold no spectral energy, so there is nothing to encode")
    coefficients = retained / norm
    min_freqs = [min_freq for min_freq, _ in bands]

    if return_fidelity:
        # The FFT is unitary up to a constant factor, so the retained fraction of the spectral
        # energy is exactly the squared overlap with the band-limited reconstruction.
        return coefficients, min_freqs, float((norm / np.linalg.norm(spectrum)) ** 2)
    return coefficients, min_freqs


@typecheck
def to_qasm(circuit: QuantumCircuit) -> str:
    """Serialize a circuit to QASM text.

    The serializer prefers QASM 2 for compatibility and falls back to QASM 3
    when QASM 2 export is not supported by the circuit.

    Args:
        circuit (QuantumCircuit): Circuit to serialize.

    Returns:
        str: QASM 2 output when possible, otherwise QASM 3 output.
    """
    try:
        return qiskit.qasm2.dumps(circuit)
    except qiskit.qasm2.QASM2ExportError:
        return qiskit.qasm3.dumps(circuit)


@typecheck
def get_circuit_hash(circuit: QuantumCircuit) -> str:
    """Compute a stable hash string for a circuit.

    The circuit is normalized before hashing by rewriting decomposable
    operations into the supported gate set (`rx`, `ry`, `rz`, `cx`) while
    leaving Haiqu-specific circuit operations intact, and by binding any
    remaining parameters to zero.

    Args:
        circuit (QuantumCircuit): Circuit to normalize and hash.

    Returns:
        str: Deterministic hash value for the normalized circuit.
    """
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


@typecheck
def find_shared_parameters(circuit: QuantumCircuit) -> list[str]:
    """Find parameter names reused across multiple operations.

    Args:
        circuit (QuantumCircuit): Parameterized circuit to inspect.

    Returns:
        list[str]: Parameter names that appear in more than one operation.
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


def job_name_from_circuits(circuits):
    """Construct a default job name from a list of circuits."""

    if not circuits:
        return None

    job_name = circuits[0].name
    num_others = len(circuits) - 1

    if num_others > 0:
        job_name += f" + {num_others} other"

        if num_others > 1:
            job_name += "s"

    return job_name


@typecheck
def get_ibmq_temporary_token(
    user_token: str,
    ibmq_api_url: str = "https://auth.quantum-computing.ibm.com/api",
) -> dict[str, Any]:
    """Exchange an IBM Quantum user token for a temporary token payload.

    The IBM Runtime API returns the temporary token under the ``id`` key. This
    helper normalizes the response by remapping that field to ``token``.

    Args:
        user_token (str): IBM Quantum API token.
        ibmq_api_url (str): IBM Quantum authentication API base URL.

    Returns:
        dict[str, Any]: Authentication response with the temporary token exposed
        as ``"token"``.
    """

    ibmq_api = Api(RetrySession(base_url=ibmq_api_url))

    response = ibmq_api.login(user_token)
    response["token"] = response.pop("id")

    return response


@typecheck
def sparse_op_to_tuple(sparse_op: SparsePauliOp) -> tuple[list[str], list[float]]:
    """Convert a sparse Pauli operator to serializable term and coefficient lists.

    The coefficients are required to be real-valued. Complex coefficients
    trigger an assertion because observables are expected to be Hermitian.

    Args:
        sparse_op (SparsePauliOp): Sparse Pauli operator to convert.

    Returns:
        tuple[list[str], list[float]]: Pauli strings and their real
        coefficients.

    Raises:
        AssertionError: If any Pauli coefficient has a non-zero imaginary part.
    """
    paulis = []
    coeffs = []

    for term, coeff in sparse_op.to_list():
        paulis.append(term)
        assert np.isclose(coeff.imag, 0.0), (
            "Observable must be a valid Hermitian operator with real coefficients. "
            f"Found {term} with complex coefficient {coeff}"
        )
        coeffs.append(coeff.real)

    return paulis, coeffs


@typecheck
def tuple_to_sparse_op(observable: tuple | list) -> SparsePauliOp:
    """Rebuild a sparse Pauli operator from its serialized term and coefficient lists.

    The inverse of :func:`sparse_op_to_tuple`, for turning the ``observables`` carried on a job
    model back into the operator the caller submitted.

    Args:
        observable (tuple | list): A ``(pauli_strings, coefficients)`` pair.

    Returns:
        SparsePauliOp: The reconstructed operator.

    Raises:
        ValueError: If the pair is malformed or the two lists have different lengths.
    """
    if len(observable) != 2:
        raise ValueError(f"Expected a (pauli_strings, coefficients) pair, got {len(observable)} elements.")

    paulis, coeffs = observable
    if len(paulis) != len(coeffs):
        raise ValueError(f"Got {len(paulis)} Pauli strings but {len(coeffs)} coefficients.")

    return SparsePauliOp(list(paulis), coeffs=list(coeffs))


def get_num_parameters_per_circuit(circuits) -> list[int | None]:
    """Return ``num_parameters`` for each circuit in a run/flow submission.

    ``QuantumCircuit`` inputs use ``circuit.num_parameters``. ``CircuitModel`` inputs use
    ``analytics.num_parameters`` when available; otherwise ``None`` (length checks are skipped).
    """
    if not isinstance(circuits, list):
        circuits = [circuits]

    nums: list[int | None] = []
    for circuit in circuits:
        if isinstance(circuit, QuantumCircuit):
            nums.append(circuit.num_parameters)
        elif getattr(circuit, "analytics", None) is not None and circuit.analytics.num_parameters is not None:
            nums.append(circuit.analytics.num_parameters)
        else:
            nums.append(None)
    return nums


def get_num_qubits_per_circuit(circuits) -> list[int | None]:
    """Return the qubit count for each circuit in a run/flow submission.

    ``QuantumCircuit`` inputs use ``circuit.num_qubits``. ``CircuitModel`` inputs use
    ``analytics.qubits`` when available; otherwise ``None`` (length checks are skipped).
    """
    if not isinstance(circuits, list):
        circuits = [circuits]

    nums: list[int | None] = []
    for circuit in circuits:
        if isinstance(circuit, QuantumCircuit):
            nums.append(circuit.num_qubits)
        elif getattr(circuit, "analytics", None) is not None and circuit.analytics.qubits is not None:
            nums.append(circuit.analytics.qubits)
        else:
            nums.append(None)
    return nums


def _is_pre_transpiled(circuit) -> bool:
    """Whether a submitted circuit has already been transpiled to a device.

    A locally transpiled ``QuantumCircuit`` carries a ``layout``; a ``CircuitModel`` returned by
    ``Haiqu.transpile`` carries a ``transpilation_target`` instead, and its ``analytics.qubits``
    reports the transpiled width rather than the logical one.
    """
    if isinstance(circuit, QuantumCircuit):
        return getattr(circuit, "layout", None) is not None
    return getattr(circuit, "transpilation_target", None) is not None


def validate_readout_bases(readout_bases, observables, circuits):
    """Validate ``readout_bases`` for a run submission.

    Checks what can be established client-side: the mutual exclusion with ``observables``, the
    shape and alphabet of the basis strings, and — only for circuits that are not already
    transpiled — that each basis covers every qubit. The authoritative width check happens
    server-side against the circuit's *active* qubits after transpilation and packing, so a
    pre-transpiled circuit (whose logical width no longer matches its active-qubit count) is
    deliberately left to the backend rather than rejected here.

    Args:
        readout_bases: The user-supplied readout bases, or None.
        observables: The user-supplied observables, or None (used only for mutual exclusion).
        circuits: The circuits being submitted, used for the width check.

    Returns:
        list[str] | None: The validated bases, unchanged, or None.

    Raises:
        ValueError: If the bases are malformed or conflict with ``observables``.
    """
    if readout_bases is None:
        return None

    if observables is not None:
        raise ValueError(
            "`observables` and `readout_bases` are mutually exclusive: observables return expectation "
            "values while readout_bases return one distribution per basis. Pass only one."
        )

    if isinstance(readout_bases, str) or not isinstance(readout_bases, (list, tuple)):
        raise ValueError(f"`readout_bases` must be a list of Pauli strings, got {type(readout_bases).__name__}.")

    readout_bases = list(readout_bases)
    if not readout_bases:
        raise ValueError("`readout_bases` must not be empty.")

    if any(not isinstance(basis, str) for basis in readout_bases):
        if all(isinstance(basis, (list, tuple)) for basis in readout_bases):
            raise ValueError(
                "Per-circuit readout bases are not supported: `readout_bases` is a flat list of Pauli "
                "strings applied to every circuit. Submit a separate job for each circuit."
            )
        raise ValueError("`readout_bases` must be a list of Pauli strings.")

    invalid = sorted({char for basis in readout_bases for char in basis} - set("XYZ"))
    if invalid:
        hint = " Use 'Z' for an unrotated qubit." if "I" in invalid else ""
        raise ValueError(f"`readout_bases` may only contain 'X', 'Y' or 'Z', got invalid characters {invalid}.{hint}")

    duplicates = sorted({basis for basis in readout_bases if readout_bases.count(basis) > 1})
    if duplicates:
        raise ValueError(
            f"`readout_bases` must not contain duplicates, got {duplicates}. Results are keyed by basis, "
            "so a repeated basis is ambiguous."
        )

    widths = {len(basis) for basis in readout_bases}
    if len(widths) > 1:
        raise ValueError(
            f"All `readout_bases` must have the same length (one character per qubit), got lengths {sorted(widths)}."
        )

    basis_width = widths.pop()
    circuits_list = circuits if isinstance(circuits, list) else [circuits]
    for index, (circuit, num_qubits) in enumerate(zip(circuits_list, get_num_qubits_per_circuit(circuits))):
        if _is_pre_transpiled(circuit):
            # Already transpiled: the basis must match the active qubits, which the backend knows.
            continue
        if num_qubits is not None and basis_width != num_qubits:
            raise ValueError(
                f"`readout_bases` entries have {basis_width} characters but circuit {index} has "
                f"{num_qubits} qubits. Each basis needs exactly one Pauli character per qubit."
            )

    return readout_bases


_PARAMETER_BINDING_ORDER_MSG = (
    "Values are bound positionally in the order of Qiskit's ``QuantumCircuit.parameters``. "
    "That order is usually alphabetical by parameter name when parameters are created one-by-one "
    "(e.g. ``theta10`` before ``theta2``), regardless of gate addition order; "
    "``ParameterVector`` and other batch constructors use vector index order instead "
    "(e.g. ``theta[0]``, ``theta[1]``) — inspect ``list(circuit.parameters)`` to confirm."
)


def validate_and_normalize_parameters_and_observables(
    parameters,
    observables,
    num_circuits,
    num_parameters_per_circuit: list[int | None] | None = None,
):
    """Validate and normalize ``parameters`` and ``observables`` for circuit submission.

    Coerces user-provided shapes into the canonical internal forms:

    - ``parameters``: 3D list of floats ``[[[..], [..]], ...]`` (one parameter group per circuit).
    - ``observables``: 3D list of ``(paulis, coeffs)`` tuples ``[[tup, ...], ...]``
      (one inner list per circuit).

    Each inner parameter list must have length ``circuit.num_parameters``. Values are bound
    positionally in the order of ``QuantumCircuit.parameters`` (Qiskit's ``ParameterView``).
    When parameters are created one-by-one, that order is typically alphabetical by name
    (independent of gate addition order); other construction patterns (e.g. ``ParameterVector``)
    use vector index order instead — use ``list(circuit.parameters)`` to verify.

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
        num_parameters_per_circuit: Optional per-circuit parameter counts used to validate inner-list
            lengths and require ``parameters`` when circuits are parameterized.

    Returns:
        Tuple ``(parameters, observables)`` in canonical internal form.

    Raises:
        ValueError: If the input shape doesn't match any accepted form.
    """

    def is_nonempty_float_list(lst):
        return isinstance(lst, list) and lst and all(isinstance(x, (int, float)) for x in lst)

    def is_nonempty_sparse_list(lst):
        return isinstance(lst, list) and lst and all(isinstance(o, SparsePauliOp) for o in lst)

    def validate_parameter_lengths():
        # Skip when counts are unavailable (None) or their length disagrees with num_circuits.
        # A length mismatch would indicate an internal bookkeeping bug rather than bad user
        # input, so fall back to the (looser) shape validation above instead of raising here.
        if num_parameters_per_circuit is None or len(num_parameters_per_circuit) != num_circuits:
            return

        for circuit_idx, expected in enumerate(num_parameters_per_circuit):
            if expected is None:
                continue
            if expected > 0 and parameters is None:
                raise ValueError(
                    f"Circuit {circuit_idx} has {expected} parameter(s) but `parameters` was not provided. "
                    f"{_PARAMETER_BINDING_ORDER_MSG}"
                )

        if parameters is None:
            return

        if num_circuits == 1:
            expected = num_parameters_per_circuit[0]
            if expected is None:
                return
            if expected == 0:
                raise ValueError("Circuit has no free parameters but `parameters` was provided.")
            for row_idx, row in enumerate(parameters):
                if len(row) != expected:
                    raise ValueError(
                        f"Parameter sweep row {row_idx} has length {len(row)}, but the circuit has "
                        f"{expected} parameter(s). Each inner list must have length equal to "
                        f"circuit.num_parameters. {_PARAMETER_BINDING_ORDER_MSG}"
                    )
            return

        for circuit_idx, group in enumerate(parameters):
            expected = num_parameters_per_circuit[circuit_idx]
            if expected is None:
                continue
            if expected == 0:
                raise ValueError(f"Circuit {circuit_idx} has no free parameters but parameter values were provided.")
            for row_idx, row in enumerate(group):
                if len(row) != expected:
                    raise ValueError(
                        f"Circuit {circuit_idx}, parameter sweep row {row_idx} has length {len(row)}, "
                        f"but the circuit has {expected} parameter(s). Each inner list must have length "
                        f"equal to circuit.num_parameters. {_PARAMETER_BINDING_ORDER_MSG}"
                    )

    # Parameters validation - accepts either 2D or 3D
    if parameters is not None:
        if not isinstance(parameters, list):
            raise ValueError("`parameters` must be a list.")

        if num_circuits == 1:
            # Single circuit: accept 2D [[p1, p2], [q1, q2], ...]
            if not all(is_nonempty_float_list(row) for row in parameters):
                raise ValueError("Single circuit: `parameters` must be 2D list of floats, " "[[p1, p2], [q1, q2], ...].")
        else:
            # Multiple circuits: accept 3D [[[c1p1, c1p2], [c1q1, c1q2]], [[c2r1, c2r2], [c2s1, c2s2]]]
            if len(parameters) != num_circuits:
                raise ValueError(f"Multiple circuits: `parameters` must have {num_circuits} entries, one per circuit.")

            if not all(all(is_nonempty_float_list(row) for row in group) for group in parameters):
                raise ValueError(
                    "Multiple circuits: `parameters` must be 3D list of floats, "
                    "[[[c1p1, c1p2], [c1q1, c1q2]], [[c2r1, c2r2], [c2s1, c2s2]]]."
                )

    validate_parameter_lengths()

    # Observables validation - always return [[tuples]] format
    if observables is not None:
        if num_circuits == 1:
            # Single circuit - always return [[tuple1, tuple2, ...]]
            if isinstance(observables, SparsePauliOp):
                # Single operator → [[tuple]]
                observables = [[sparse_op_to_tuple(observables)]]
            elif isinstance(observables, list) and len(observables) == 1 and is_nonempty_sparse_list(observables[0]):
                # Nested form [[op1, op2, ...]] → [[tuple1, tuple2, ...]]
                observables = [[sparse_op_to_tuple(op) for op in observables[0]]]
            elif is_nonempty_sparse_list(observables):
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
                elif is_nonempty_sparse_list(circuit_obs):
                    # List of operators for this circuit → [tuple1, tuple2, ...]
                    processed.append([sparse_op_to_tuple(op) for op in circuit_obs])
                else:
                    raise ValueError(
                        f"Circuit {circuit_idx}: Expected SparsePauliOp or list of SparsePauliOp objects, "
                        f"but received {type(circuit_obs).__name__}: {circuit_obs}"
                    )
            observables = processed

    return parameters, observables


@typecheck
def is_haiqu_generated(circuit: QuantumCircuit) -> bool:
    """Check whether a circuit contains Haiqu-generated operations.

    Args:
        circuit (QuantumCircuit): Circuit to inspect.

    Returns:
        bool: ``True`` if the circuit contains a Haiqu operation, otherwise
        ``False``.
    """
    for instruction in circuit.data:
        if instruction.operation.name in HAIQU_OPERATIONS:
            return True
    return False


def preprocess_metrics(**kwargs: Any) -> dict[str, Any]:
    """Normalize user-defined metrics into JSON-friendly values.

    Drawer plots, Matplotlib figures, and Pandas data frames are converted into
    serialized forms that can be sent to the API. Other values are passed
    through unchanged.

    Args:
        **kwargs (Any): Arbitrary metric names and values.

    Returns:
        dict[str, Any]: Dictionary of normalized metric values ready for
        transport.
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
    """Generate an artifact name from the provided context.

    The prefix is inferred from common object types such as Drawer plots,
    Matplotlib figures, and Pandas data frames.

    Args:
        parent_ctx (Any): Primary object being logged.
        child_ctx (Any): Secondary object associated with the log entry.

    Returns:
        str: Timestamped artifact name with a type-appropriate prefix.
    """
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


def check_circuit_context(ctx: Any) -> QuantumCircuit | None:
    """Interpret a logging context as a quantum circuit when possible.

    String inputs are treated as potential QASM payloads and parsed through
    ``ensure_qc()``. Invalid QASM strings are treated as non-circuit inputs.

    Args:
        ctx (Any): Context object to inspect.

    Returns:
        QuantumCircuit | None: Parsed ``QuantumCircuit`` when the input
        represents a circuit; otherwise ``None``.
    """
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


def _ipython_user_ns() -> dict:
    """Return the IPython user namespace, or an empty dict when unavailable."""
    try:
        from IPython import get_ipython

        ipy = get_ipython()
        if ipy is not None:
            return ipy.user_ns
    except Exception:
        pass
    return {}


def _kernel_id() -> str | None:
    """Return the id of the running IPython kernel, or ``None``.

    The kernel connection file is named ``kernel-<id>.json``; the ``<id>``
    matches the ``kernel.id`` reported by the Jupyter server sessions API.
    """
    try:
        from ipykernel.connect import get_connection_file

        name = os.path.basename(get_connection_file())
        if name.startswith("kernel-") and name.endswith(".json"):
            return name[len("kernel-") : -len(".json")]  # noqa: E203
    except Exception:
        pass
    return None


def _notebook_path_from_server() -> str | None:
    """Return the notebook's absolute path from the live Jupyter server.

    Queries each running Jupyter server's ``/api/sessions`` endpoint and matches
    the current kernel. This reflects notebook renames immediately, unlike the
    ``JPY_SESSION_NAME`` environment variable which is fixed when the kernel
    starts.

    Returns ``None`` when no session matches, on any error, or when the kernel is
    shared by more than one notebook — in that ambiguous case there is no single
    correct path, so we decline to guess and let the caller fall back.
    """
    kernel_id = _kernel_id()
    if kernel_id is None:
        return None

    try:
        from jupyter_server.serverapp import list_running_servers
    except Exception:
        return None

    matches = set()
    for server in list_running_servers():
        try:
            token = server.get("token", "") or ""
            resp = requests.get(
                urljoin(server["url"], "api/sessions"),
                headers={"Authorization": f"token {token}"} if token else {},
                timeout=2,
            )
            resp.raise_for_status()
            sessions = resp.json()

            for sess in sessions:
                if sess.get("kernel", {}).get("id") != kernel_id:
                    continue
                rel_path = (sess.get("notebook") or {}).get("path") or sess.get("path")
                if not rel_path:
                    continue
                root_dir = server.get("root_dir") or server.get("notebook_dir") or ""
                matches.add(os.path.join(root_dir, rel_path) if root_dir else rel_path)
        except Exception:
            continue

    # A single unambiguous match is the notebook; anything else is not safe to guess.
    return matches.pop() if len(matches) == 1 else None


def _notebook_full_path() -> str | None:
    """Return the absolute path of the running Jupyter notebook, or ``None``.

    Sources are tried in order of reliability:

    1. The live Jupyter server sessions API, which reflects the current name
       even after a rename (JupyterLab/Notebook web app).
    2. ``__vsc_ipynb_file__``, injected by the VS Code Jupyter extension for
       local notebooks.
    3. The ``__session__`` variable / ``JPY_SESSION_NAME`` environment variable,
       which are fixed when the kernel starts (and thus stale after a rename).

    Example: ``/user_storage/examples/run_examples/200_GHZStatePreparation.ipynb``.
    """
    from_server = _notebook_path_from_server()
    if from_server:
        return from_server

    user_ns = _ipython_user_ns()

    vscode_path = user_ns.get("__vsc_ipynb_file__")
    if vscode_path:
        return vscode_path

    return user_ns.get("__session__") or os.environ.get("JPY_SESSION_NAME")


def detect_notebook() -> str | None:
    """Detect whether execution is happening inside a Jupyter notebook.

    Returns:
        str | None: Notebook path when it can be determined, otherwise
        ``None``. Example: ``/examples/run_examples/200_GHZStatePreparation.ipynb``.
    """
    if not is_jupyter():
        return

    full_path = _notebook_full_path()

    if full_path is None:
        return

    bits = full_path.split("user_storage")
    if len(bits) != 2:
        # Unable to parse notebook path, return full path as is
        # Let Dashboard handle it (useful for local runs)
        return full_path

    return bits[1]


def detect_source_file() -> str | None:
    """Detect the absolute on-disk path of the running notebook or script.

    Unlike :func:`detect_notebook`, which returns a display path for the
    Dashboard, this returns the actual filesystem path so the file's content
    can be read. Works both inside a Jupyter notebook and for a plain
    ``python script.py`` invocation.

    Returns:
        str | None: Absolute path of the notebook or script, or ``None`` when it
        cannot be determined (e.g. an interactive REPL or ``python -c``).
    """
    if is_jupyter():
        return _notebook_full_path()

    # Plain script execution: ``sys.argv[0]`` is the entry-point script.
    argv0 = sys.argv[0] if sys.argv else ""
    if not argv0:
        return None

    abspath = os.path.abspath(argv0)
    if abspath.endswith(".py") and os.path.isfile(abspath):
        return abspath

    return None


def _strip_notebook_outputs(raw: str) -> str:
    """Drop cell outputs from a notebook's JSON to shrink the logged payload.

    Only code-cell ``outputs`` and ``execution_count`` are cleared; source is
    preserved. Returns the input unchanged if it is not parseable notebook JSON.

    Args:
        raw (str): Raw ``.ipynb`` file content.

    Returns:
        str: Notebook JSON with outputs stripped, or the original ``raw``.
    """
    try:
        notebook = json.loads(raw)
        for cell in notebook.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
        return json.dumps(notebook)
    except Exception:
        return raw


def read_source_file(path: str) -> str | None:
    """Read notebook/script content for logging, best-effort.

    Notebook outputs are stripped before sizing. Content exceeding
    :data:`~haiqu.sdk.constants.MAX_SOURCE_FILE_SIZE` (after stripping) is
    skipped; in that case ``None`` is returned and no source content is logged.
    Never raises: any read/parse error yields ``None``.

    Args:
        path (str): Absolute path returned by :func:`detect_source_file`.

    Returns:
        str | None: File content ready to log, or ``None`` when unavailable or
        too large.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return None

    if path.endswith(".ipynb"):
        content = _strip_notebook_outputs(content)

    if len(content.encode("utf-8")) > MAX_SOURCE_FILE_SIZE:
        return None

    return content


@typecheck
def setup_logger(logger_name: str, log_level: int = logging.INFO) -> logging.Logger:
    """Create and configure a stream logger.

    The logger and its stream handler both use the provided log level and a
    consistent SDK log format.

    Args:
        logger_name (str): Logger name.
        log_level (int): Logging level applied to the logger and stream handler.

    Returns:
        logging.Logger: Configured logger instance.
    """
    LOG_FORMAT = "%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"

    logger = logging.getLogger(logger_name)
    logger.setLevel(log_level)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter(LOG_FORMAT)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def validate_error_mitigation_options(options: dict, use_mitigation: bool) -> None:
    """Validate the optional ``error_mitigation_options`` entry of an ``options`` dict.

    Shared by ``Haiqu.run`` and ``Haiqu.variational_optimization`` so both expose the same
    granular error-mitigation controls. Only the key names and value types are validated here;
    the per-component default values are applied server-side (and differ between the run and
    variational pipelines), so this function is intentionally default-agnostic.

    Args:
        options (dict): The user-provided options dict (may or may not contain
            ``error_mitigation_options``).
        use_mitigation (bool): Whether mitigation is enabled for this job. Used only to warn when
            mitigation options are supplied but would have no effect.

    Raises:
        ValueError: If ``error_mitigation_options`` is not a dict, contains unknown keys, or has a
            value of the wrong type for a known key.
    """
    if "error_mitigation_options" not in options:
        return

    emo = options["error_mitigation_options"]
    if not isinstance(emo, dict):
        raise ValueError("`error_mitigation_options` in `options` must be a dictionary.")
    dict_type_keys = {"readout_mitigation_options"}
    bool_type_keys = {
        "dynamical_decoupling",
        "readout_mitigation",
        "noise_tailoring",
        "advanced_mitigation",
    }
    valid_emo_keys = dict_type_keys | bool_type_keys
    unknown_keys = set(emo.keys()) - valid_emo_keys
    if unknown_keys:
        raise ValueError(f"Unknown key(s) in `error_mitigation_options`: {unknown_keys}. " f"Valid keys are: {valid_emo_keys}.")
    for key, val in emo.items():
        if key in dict_type_keys:
            if not isinstance(val, dict):
                raise ValueError(f"{key} must be a `dict`. Got {type(val).__name__!r} for key '{key}'.")
        elif key in bool_type_keys:
            if not isinstance(val, bool):
                raise ValueError(f"{key} must be a `bool`. Got {type(val).__name__!r} for key '{key}'.")
    if not use_mitigation:
        warnings.warn(
            "`error_mitigation_options` provided but `use_mitigation=False`. "
            "Mitigation options will have no effect unless `use_mitigation=True`.",
            UserWarning,
            stacklevel=3,
        )
    if not emo.get("readout_mitigation", False) and emo.get("readout_mitigation_options", None):
        warnings.warn(
            "`readout_mitigation_options` provided but `readout_mitigation=False`. "
            "Readout mitigation options will have no effect unless `readout_mitigation=True`.",
            UserWarning,
            stacklevel=3,
        )
