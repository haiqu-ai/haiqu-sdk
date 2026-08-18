"""Layers that make up a hybrid program.

A program is an ordered list of layers describing how your circuits are processed
and run. It starts with an :class:`InputLayer` and ends with a :class:`DeviceLayer`.

Use :class:`EstimatorLayer` or :class:`DistributionMitigationLayer` for grouped
error mitigation, or compose finer processing steps by hand. For advanced
mitigation in manual pipelines, pick the layer for the mitigation path:
:class:`AdvancedObsMitigationLayer` for observable-based mitigation and
:class:`AdvancedDistMitigationLayer` for distribution-based mitigation.
"""
from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class InputLayer(BaseModel):
    """The program's entry point. Every program starts with one."""

    type: Literal["input"] = "input"


class DeviceLayer(BaseModel):
    """Runs the circuits on a backend; every program ends with one.

    ``device_id`` selects the backend; ``options`` carries backend-specific
    settings (e.g. credentials). For real IBM QPUs, pass
    ``options={"use_fractional_gates": True}`` to use fractional gates
    (continuous-angle ``rx`` / ``rzz``, etc.). Outside hybrid programs, set the
    same flag on :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.get_device` instead.
    """

    type: Literal["device"] = "device"

    device_id: str
    options: dict = {}


class PackingLayer(BaseModel):
    """Pack several copies of a circuit into one run to use spare device qubits.

    ``pack_size`` is the number of copies; leave it unset to pick a value
    automatically from the circuit and device sizes.
    """

    type: Literal["packing"] = "packing"

    pack_size: Annotated[int, Field(ge=2)] | None = None


class TranspilationLayer(BaseModel):
    """Transpile the circuits for the target backend.

    ``optimization_level`` (0-3) sets the optimization effort; leave it unset for
    the default.
    """

    type: Literal["transpilation"] = "transpilation"

    optimization_level: Literal[0, 1, 2, 3] | None = None


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


Layer = Annotated[
    Union[
        InputLayer,
        DeviceLayer,
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
    ],
    Field(discriminator="type"),
]
