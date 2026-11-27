"""Layers that make up a hybrid program.

A program is an ordered list of layers describing how your circuits are processed
and run. It starts with an :class:`InputLayer` and ends with the layer that produces
its results: a :class:`DeviceLayer` to run on a backend, a :class:`StatevectorLayer`
to simulate exactly instead, a :class:`CliffordTSynthesisLayer` to approximate the
circuits with Clifford+T gates and return them without running them, or a
:class:`QECSamplerLayer` to sample circuits compiled by a :class:`QECCompilationLayer`
to a quantum error-correcting code under noise.

Use :class:`EstimatorLayer` or :class:`DistributionMitigationLayer` for grouped
error mitigation, or compose finer processing steps by hand. For advanced
mitigation in manual pipelines, pick the layer for the mitigation path:
:class:`AdvancedObsMitigationLayer` for observable-based mitigation and
:class:`AdvancedDistMitigationLayer` for distribution-based mitigation.
"""
import os
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


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


class CliffordTSynthesisLayer(BaseModel):
    """Approximate the circuits' non-Clifford rotations with Clifford+T gates, and return them.

    An alternative ending to a program: use this instead of a :class:`DeviceLayer` when you
    want the circuits approximated with Clifford+T gates rather than run. Nothing is executed,
    so the job's ``shots`` is not used and no device time is spent. A program ending here has
    no other layer after its :class:`InputLayer`.

    Each rotation outside the Clifford group is replaced by a sequence of Clifford and T
    gates that approximates it. How closely is set by one of four options; when several are
    set, the first of them in this order is used and the rest are ignored:
    ``total_t_gate_budget``, ``gate_t_gate_budget``, ``total_precision_tolerance``,
    ``gate_precision_tolerance``. With none set, the whole circuit is kept within a precision
    tolerance of ``1e-8``. The precision of one rotation is the spectral norm distance to its
    approximation, minimized over global phase; that of the whole circuit is the sum of these
    distances.

    A precision tolerance is met with GridSynth, a T-gate budget with TRASYN, which tends to
    use fewer T gates but takes longer the larger the budget. A per-gate budget above
    ``max_trasyn_budget`` is met with GridSynth instead. With ``exact_spectral_norm`` set, the
    synthesis also reports the exact spectral norm distance between each whole circuit and its
    approximation; it is computed from their full unitaries, so it is only available for
    circuits of up to 10 qubits. See :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.clifford_t_synthesis`
    for details.

    The job returns one :class:`~haiqu.sdk.schemas.CircuitModel` per input circuit, in input
    order, each logged to the job's experiment as a child of its input circuit. Final
    measurements are kept. The job's ``info["clifford_t_synthesis"]`` holds what the synthesis
    reports for each input circuit.
    """

    type: Literal["clifford_t_synthesis"] = "clifford_t_synthesis"

    #: T-gate budget for the whole circuit.
    total_t_gate_budget: Annotated[int, Field(ge=0)] | None = None
    #: T-gate budget for each synthesized rotation.
    gate_t_gate_budget: Annotated[int, Field(ge=0)] | None = None
    #: Precision tolerance for the whole circuit; at least ``1e-13``.
    total_precision_tolerance: Annotated[float, Field(ge=1e-13)] | None = None
    #: Precision tolerance for each synthesized rotation; at least ``1e-13``.
    gate_precision_tolerance: Annotated[float, Field(ge=1e-13)] | None = None
    #: Largest per-gate T-gate budget met with TRASYN rather than GridSynth, from 0 to 65.
    #: Unset means the synthesis default, 13.
    max_trasyn_budget: Annotated[int, Field(ge=0, le=65)] | None = None
    #: Report the exact spectral norm distance of each whole circuit (up to 10 qubits).
    exact_spectral_norm: bool = False


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


# --- Encoding ------------------------------------------------------------------


class DFSEncodingLayer(BaseModel):
    """Encode every qubit into a three-qubit decoherence-free subsystem (DFS-3).

    Each qubit of the circuit is carried by three physical qubits arranged so that noise acting
    identically on all three leaves the encoded information untouched. The encoded circuit is
    considerably deeper than the original and needs three times as many qubits on the device;
    results are reported over the original (logical) qubits.

    ``postselect`` (default ``True``) discards shots in which an encoded qubit left its protected
    state; the discarded fraction is reported in ``job.dfs_discard_fraction`` (distribution
    jobs only).
    ``synthesize_unknown`` lets gates without a built-in encoded version be synthesized
    numerically, which is slow; by default such gates are rejected.

    Place it before the :class:`TranspilationLayer`. To measure observables, put
    :class:`ObservableSplitLayer` and :class:`QWCComputeLayer` before it (the grouped
    :class:`EstimatorLayer` cannot be combined with it). Cannot be combined with :class:`PackingLayer`
    or :class:`StatevectorLayer`.
    These placement rules are checked when the program is compiled for execution.

    Device connectivity matters: the three physical qubits of an encoded qubit should sit next
    to each other on the device. On devices that are not all-to-all connected the transpiler may
    spread them apart, which adds routing overhead to every gate and weakens the protection (and
    disables :class:`SwapDDLayer` for that qubit). Check the transpiled circuit, or use a device
    whose layout keeps each triple together.
    """

    type: Literal["dfs_encoding"] = "dfs_encoding"

    postselect: bool = True
    synthesize_unknown: bool = False


class SwapDDLayer(BaseModel):
    """Swap dynamical decoupling for circuits encoded with :class:`DFSEncodingLayer`.

    While an encoded qubit idles, its three physical qubits are cycled through all their role
    permutations with physical SWAP gates, which averages out slowly varying noise. Requires a
    :class:`DFSEncodingLayer` earlier in the program and must come after the
    :class:`TranspilationLayer`; place it before :class:`DynamicalDecouplingLayer` or a grouped
    mitigation layer. Swap sequences are only inserted on devices that expose gate durations.

    ``fix_total=False`` (default) adds the swap durations on top of each idle period;
    ``fix_total=True`` keeps each idle period's total duration fixed and skips periods too short
    for the swaps. ``insert_barriers`` keeps the swap cycle from being reordered. ``max_dwell_dt``
    caps how long (in device time steps) a block stays in one permutation, so longer idle periods
    get more cycles; unset, one cycle spans the whole period. ``strict`` fails the job instead of
    warning when an encoded qubit cannot be protected, or when no swap sequences can be inserted
    at all (for example on a simulator without gate durations). How many encoded qubits were
    protected is reported in ``job.swap_dd_coverage``.

    Requires each encoded qubit's three physical qubits to be mutually connected on the device;
    on devices that are not all-to-all connected, a qubit whose triple was spread apart by the
    transpiler is left undecorated. The swaps must not be re-routed afterwards, since the scheme
    relies on equal time spent in each permutation.
    """

    type: Literal["swap_dd"] = "swap_dd"

    fix_total: bool = False
    insert_barriers: bool = True
    max_dwell_dt: Annotated[int, Field(ge=1)] | None = None
    strict: bool = False


# --- Quantum error correction --------------------------------------------------
# A program with these layers is exactly InputLayer -> QECCompilationLayer -> QECSamplerLayer.


class QECCompilationLayer(BaseModel):
    """Compile the circuits to a quantum error-correcting code, as
    :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.qec_compile` does.

    Each circuit is a logical circuit: each of its qubits is encoded into the code, and
    error-correction cycles are inserted between its logical gates. ``code`` is a code of the Haiqu
    QEC code library, given by its slug (e.g. ``"steane_7_1_3"``) or as a code returned by
    ``haiqu.get_qec_code``, or a custom code, given by the path to its QECX file or by the slug it was
    stored under: :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.flow` stores the file in the current experiment,
    as ``qec_compile`` does, and submits the custom code's slug in its place. As in ``qec_compile``, a
    string is read as a path when it names an existing file or contains ``/`` or ``\\``. A slug that names
    no code makes the job fail. A file opened in binary mode is not accepted here; pass its path.

    The options are those of ``qec_compile``; one left unset takes the default described there:
    ``ec_cycle_frequency`` (an error-correction cycle after every ``N`` layers that contain a logical
    gate, ``0`` for none), ``variant_selections`` (the code's variants to use for its operations, by
    name) and ``add_layer_barriers`` (a barrier between consecutive logical layers).

    The compiled circuits are sampled by a :class:`QECSamplerLayer`, which must follow directly: a
    program with QEC layers is exactly ``InputLayer``, ``QECCompilationLayer``, ``QECSamplerLayer``.
    This is checked when the program is created. To get the compiled circuits themselves, use
    ``qec_compile``.
    """

    type: Literal["qec_compilation"] = "qec_compilation"

    code: Annotated[str, Field(min_length=1, max_length=1024)]
    ec_cycle_frequency: Annotated[int, Field(ge=0)] | None = None
    variant_selections: dict | None = None
    add_layer_barriers: bool | None = None

    @field_validator("code", mode="before")
    @classmethod
    def _code_as_string(cls, code):
        # The schemas module imports this one, so the code model of qec_compile is imported here.
        from ..schemas import QECCodeModel

        if isinstance(code, QECCodeModel):
            return code.slug
        if isinstance(code, os.PathLike):
            # Its string form can read as a slug, so a missing file is refused here rather than at the server
            if not os.path.isfile(code):
                raise ValueError(f"QECX file '{os.fspath(code)}' does not exist")
            return os.fspath(code)
        return code


class QECSamplerLayer(BaseModel):
    """Sample the QEC-compiled circuits under noise and decode the outcomes, as
    :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.qec_run` does.

    An alternative ending to a program, directly after a :class:`QECCompilationLayer`. Nothing runs
    on a device: each compiled circuit is sampled ``shots`` times under the noise model ``noise``,
    the syndrome measured in its error-correction cycles is decoded, and the outcomes are read in the
    logical frame. The options are those of ``qec_run``, with the same defaults: ``noise`` (required,
    a dict with the keys ``"p1q"``, ``"p2q"`` and ``"p_meas"``, or a
    :class:`~haiqu.sdk.schemas.QECNoiseModel`), ``sampler``, ``decoder``, ``seed`` and ``toggles``.
    With the same circuits, code, options, shots and seed, a program ending here gives the same counts
    as ``qec_run``.

    The job returns one measurement distribution per input circuit over the logical circuit's
    classical bits, the decoded counts divided by ``shots``. ``job.qec_results`` holds the decoded
    counts, the logical error rate and the decoder used for each circuit, as ``qec_run`` reports them.
    Observables, readout bases, parameter sweeps, device credentials, dry runs and circuits that are
    already QEC-compiled do not apply, and :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.flow` refuses them.
    """

    type: Literal["qec_sampler"] = "qec_sampler"

    noise: dict[str, float]
    sampler: Literal["stim"] = "stim"
    decoder: Literal["auto", "pymatching", "chromobius", "bposd", "tesseract"] = "auto"
    seed: Annotated[int, Field(ge=0)] | None = None
    toggles: dict | None = None

    @field_validator("noise", mode="before")
    @classmethod
    def _check_noise(cls, noise):
        # The schemas module imports this one, so the noise model of qec_run is imported here.
        from ..schemas import QECNoiseModel

        return QECNoiseModel.model_validate(noise).model_dump()


Layer = Annotated[
    Union[
        InputLayer,
        DeviceLayer,
        StatevectorLayer,
        CliffordTSynthesisLayer,
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
        DFSEncodingLayer,
        SwapDDLayer,
        QECCompilationLayer,
        QECSamplerLayer,
    ],
    Field(discriminator="type"),
]
