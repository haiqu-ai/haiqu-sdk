"""Layers that make up a hybrid program.

A program is an ordered list of layers describing how your circuits are processed
and run. It starts with an :class:`InputLayer` and ends with the layer that produces
its results: a :class:`DeviceLayer` to run on a backend, or a
:class:`StatevectorLayer` to simulate exactly instead.

Use :class:`EstimatorLayer` or :class:`DistributionMitigationLayer` for grouped
error mitigation, or compose finer processing steps by hand. For advanced
mitigation in manual pipelines, pick the layer for the mitigation path:
:class:`AdvancedObsMitigationLayer` for observable-based mitigation and
:class:`AdvancedDistMitigationLayer` for distribution-based mitigation.
"""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


class InputLayer(BaseModel):
    """The program's entry point. Every program starts with one."""

    type: Literal["input"] = "input"


class DeviceLayer(BaseModel):
    """Runs the circuits on a backend.

    ``device_id`` selects the backend; ``options`` carries backend-specific settings.
    For real IBM QPUs, pass ``options={"use_fractional_gates": True}`` to use fractional gates
    (continuous-angle ``rx`` / ``rzz``, etc.). Outside hybrid programs, set the
    same flag on :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.get_device` instead.
    """

    type: Literal["device"] = "device"

    device_id: Annotated[str, Field(max_length=200)]
    options: dict = {}
    auto_quality: bool = False


class StatevectorLayer(BaseModel):
    """Simulate the circuits exactly and return their statevectors.

    An alternative ending to a program: use this instead of a
    :class:`DeviceLayer` when you want the exact state rather than measurement
    results. There is no device and no sampling, so the job's ``shots`` is not
    used and the result carries no shot noise.

    The job returns one complex-valued statevector per input circuit, each of
    length ``2 ** num_qubits`` in standard Qiskit ordering (rightmost bit =
    qubit 0). Final measurements in the input circuits are ignored; circuits
    that measure part-way through have no statevector and are rejected.

    Exact simulation cost doubles with every qubit, so the circuit size is
    capped; observables and readout bases do not apply, and neither do the
    mitigation, packing and observable layers -- there is no noise to mitigate
    and no measurement to interpret. A program with one of those layers is
    rejected when it is submitted.
    """

    type: Literal["statevector"] = "statevector"


class PackingLayer(BaseModel):
    """Pack several copies of a circuit into one run to use spare device qubits.

    ``pack_size`` is the number of copies; leave it unset to pick a value
    automatically from the circuit and device sizes.
    """

    type: Literal["packing"] = "packing"

    pack_size: Annotated[int, Field(ge=2)] | None = None


class TranspilationLayer(BaseModel):
    """Transpile the circuits for the target backend.

    Every field is optional; one left unset is not passed to the transpiler at all, so
    Qiskit's own default applies.

    Options not listed here are accepted and forwarded to the transpiler as given, so a
    Qiskit option this layer has not caught up with can still be used. The trade-off is that
    a misspelled name is forwarded too.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["transpilation"] = "transpilation"

    #: Optimization effort. Unset means the Haiqu default rather than Qiskit's.
    optimization_level: Literal[0, 1, 2, 3] | None = None

    basis_gates: list[str] | None = None
    coupling_map: list[list[int]] | None = None
    initial_layout: list[int] | None = None
    layout_method: Literal["trivial", "dense", "sabre"] | None = None
    routing_method: Literal["basic", "lookahead", "sabre", "none"] | None = None
    translation_method: Literal["translator", "synthesis"] | None = None
    scheduling_method: Literal["alap", "asap"] | None = None
    approximation_degree: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    seed_transpiler: int | None = None
    dt: float | None = None
    qubits_initially_zero: bool | None = None


# --- Error mitigation --------------------------------------------------------
# A program uses at most one of these. Choose EstimatorLayer when you measure
# observables (expectation values), DistributionMitigationLayer when you read the
# raw measurement distribution. The flags turn individual techniques on or off.


class EstimatorLayer(BaseModel):
    """Measure observable expectation values with error mitigation.

    Use this when the job supplies observables. Error mitigation is opt-in.
    """

    type: Literal["estimator"] = "estimator"

    mitigation_enabled: bool = False
    advanced_mitigation: bool = True
    readout_mitigation: bool = True
    noise_tailoring: bool = False
    dynamical_decoupling: bool = True

    readout_mitigation_options: dict = {}


class DistributionMitigationLayer(BaseModel):
    """Mitigate errors on the raw measured probability distribution.

    Use this when the job reads measurement outcomes (no observables).
    """

    type: Literal["distribution_mitigation"] = "distribution_mitigation"

    mitigation_enabled: bool = True
    advanced_mitigation: bool = True
    readout_mitigation: bool = True
    noise_tailoring: bool = False
    dynamical_decoupling: bool = True

    readout_mitigation_options: dict = {}


# Finer-grained steps for hand-built pipelines; advanced mitigation splits by mitigation path.


class ObservableSplitLayer(BaseModel):
    """Split a task with several observables into one task per observable."""

    type: Literal["observable_split"] = "observable_split"


class NoiseTailoringLayer(BaseModel):
    """Tailor device noise with Pauli twirling."""

    type: Literal["noise_tailoring"] = "noise_tailoring"


class DynamicalDecouplingLayer(BaseModel):
    """Suppress idle-qubit errors with dynamical-decoupling sequences."""

    type: Literal["dynamical_decoupling"] = "dynamical_decoupling"


class AdvancedObsMitigationLayer(BaseModel):
    """Advanced observable-based error mitigation for hand-built pipelines.

    Use this in place of the ``advanced_mitigation`` flag on grouped mitigation
    layers when manually enabling observable-based advanced mitigation.
    """

    type: Literal["advanced_obs_mitigation"] = "advanced_obs_mitigation"


class AdvancedDistMitigationLayer(BaseModel):
    """Advanced distribution-based error mitigation for hand-built pipelines.

    Use this in place of the ``advanced_mitigation`` flag on grouped mitigation
    layers when manually enabling distribution-based advanced mitigation.
    """

    type: Literal["advanced_dist_mitigation"] = "advanced_dist_mitigation"


class AdvancedReadoutMitigationLayer(BaseModel):
    """Advanced measurement (readout) error mitigation."""

    type: Literal["advanced_readout_mitigation"] = "advanced_readout_mitigation"


class QWCComputeLayer(BaseModel):
    """Compute observable expectation values from grouped commuting measurements."""

    type: Literal["qwc_compute"] = "qwc_compute"


class ReadoutBasisLayer(BaseModel):
    """Read the circuits out in the per-qubit Pauli bases carried by the job submission.

    The bases are a job-level field (``readout_bases``), not a layer field. This layer marks where the rotation is applied during
    execution of the hybrid program, and the result depends on the placement relative to the :class:`TranspilationLayer`:

    - After the :class:`TranspilationLayer`: all bases of a circuit share one transpilation and one layout.
    - Before the :class:`TranspilationLayer`: each basis is transpiled separately.
    - No :class:`TranspilationLayer`: the rotation applies to the pre-transpiled circuit's active qubits.

    To ensure that each basis is twirled independently rather than sharing one sample, place the rotation prior to
    :class:`NoiseTailoringLayer`.

    Cannot be combined with an :class:`EstimatorLayer`.
    """

    type: Literal["readout_basis"] = "readout_basis"


Layer = Annotated[
    Union[
        InputLayer,
        DeviceLayer,
        StatevectorLayer,
        PackingLayer,
        TranspilationLayer,
        EstimatorLayer,
        DistributionMitigationLayer,
        ObservableSplitLayer,
        NoiseTailoringLayer,
        DynamicalDecouplingLayer,
        AdvancedObsMitigationLayer,
        AdvancedDistMitigationLayer,
        AdvancedReadoutMitigationLayer,
        QWCComputeLayer,
        ReadoutBasisLayer,
    ],
    Field(discriminator="type"),
]
