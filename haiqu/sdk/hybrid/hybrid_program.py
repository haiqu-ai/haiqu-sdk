from pydantic import BaseModel, PositiveInt, field_validator

from .layers import (
    CliffordTSynthesisLayer,
    DeviceLayer,
    DistributionMitigationLayer,
    EstimatorLayer,
    InputLayer,
    Layer,
    StatevectorLayer,
)

# Grouped mitigation variants are mutually exclusive: a program is either an
# observable-based (estimator) or distribution-based mitigation path, never both.
_GROUPED_MITIGATION = (EstimatorLayer, DistributionMitigationLayer)

# A layer that produces the program's results, and so ends it. A program has exactly one.
_TERMINAL = (DeviceLayer, StatevectorLayer, CliffordTSynthesisLayer)


class HybridProgram(BaseModel):
    schema_version: PositiveInt = 1

    layers: list[Layer]

    @field_validator("schema_version", mode="before")
    @classmethod
    def _check_schema_version(cls, schema_version):
        if schema_version != 1:
            raise ValueError(f"schema_version must be 1, not {schema_version}")

        return schema_version

    @field_validator("layers")
    @classmethod
    def _check_layers(cls, layers):
        if not layers:
            raise ValueError("layers must not be empty")
        elif not isinstance(layers[0], InputLayer):
            raise ValueError("First layer must be InputLayer")
        elif any(isinstance(layer, InputLayer) for layer in layers[1:-1]):
            raise ValueError("Only one InputLayer is allowed")
        elif not isinstance(layers[-1], _TERMINAL):
            raise ValueError("Last layer must be DeviceLayer, StatevectorLayer or CliffordTSynthesisLayer")
        elif any(isinstance(layer, _TERMINAL) for layer in layers[1:-1]):
            raise ValueError("Only one DeviceLayer, StatevectorLayer or CliffordTSynthesisLayer is allowed")

        if isinstance(layers[-1], CliffordTSynthesisLayer) and len(layers) > 2:
            others = ", ".join(type(layer).__name__ for layer in layers[1:-1])
            raise ValueError(
                "A program ending in CliffordTSynthesisLayer has nothing between its InputLayer and "
                f"CliffordTSynthesisLayer, not {others}"
            )

        grouped = [layer for layer in layers if isinstance(layer, _GROUPED_MITIGATION)]
        if len(grouped) > 1:
            raise ValueError(
                "At most one grouped mitigation layer (estimator / "
                "distribution_mitigation) is allowed; they are mutually exclusive"
            )

        return layers
