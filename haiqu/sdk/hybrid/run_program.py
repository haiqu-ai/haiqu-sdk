"""Build the HybridProgram for a haiqu.run() call.

The `options` from haiqu.run() is split three ways:
* keys that describe the *pipeline* become layers (packing, transpilation, mitigation);
* keys that are device credentials travel in the submission's own ``device_credentials``
  field, never inside the program;
* everything else is a backend constructor argument and rides on the device layer.
"""

from __future__ import annotations

import copy
from typing import Any, Optional

from .hybrid_program import HybridProgram
from . import layers

ALLOWED_DEVICE_CREDENTIALS = frozenset(
    {
        # AWS
        "aws_access_key_id",
        "aws_secret_access_key",
        "aws_default_region",
        "aws_session_token",
        # IBM
        "ibm_quantum_token",
        "ibm_quantum_instance",
        # IonQ
        "ionq_api_key",
        # IQM
        "iqm_token",
        "iqm_server_url",
    }
)


# Mitigation defaults for a `haiqu.run` submission
RUN_MITIGATION_DEFAULTS = {
    "noise_tailoring": True,
    "readout_mitigation": False,
}


def program_from_run_options(
    *,
    device_id: str,
    options: Optional[dict] = None,
    use_mitigation: bool = False,
    use_packing: bool = False,
    pack_size: Optional[int] = None,
    has_observables: bool = False,
    auto_quality: bool = False,
) -> tuple[HybridProgram, dict[str, Any]]:
    """Translate haiqu.run() arguments into a hybrid program.

    Args:
        device_id: The device the program executes on. Becomes the terminal device layer.
        options: haiqu.run() options dict. Not mutated -- it is copied first.
        use_mitigation: Turns on the grouped mitigation layer. Always passed through
            explicitly, because the two grouped layers ship opposite defaults.
        use_packing: Adds a packing layer.
        pack_size: The packing layer's size. ``None`` means let the server resolve it from
            the device and circuit geometry.
        has_observables: Whether the submission carries observables, which selects the
            estimator branch over the distribution branch.
        auto_quality: Whether to assess the results against an ideal simulation.

    Returns:
        The program, and the device credentials lifted out of ``options``.

    """
    options = {} if options is None else copy.deepcopy(options)
    device_credentials = {key: options.pop(key) for key in list(options) if key in ALLOWED_DEVICE_CREDENTIALS}

    skip_transpilation = bool(options.pop("skip_transpilation", False))
    error_mitigation_options = options.pop("error_mitigation_options", None) or {}
    transpilation_options = options.pop("transpilation_options", None) or {}

    # Accepted in `options` too, for callers who set it before it became an argument.
    auto_quality = bool(options.pop("auto_quality", False)) or auto_quality

    program_layers: list[Any] = [layers.InputLayer()]

    if use_packing:
        program_layers.append(layers.PackingLayer(pack_size=pack_size))

    if not skip_transpilation:
        program_layers.append(layers.TranspilationLayer(**transpilation_options))

    mitigation_flags = {
        "mitigation_enabled": bool(use_mitigation),
        **RUN_MITIGATION_DEFAULTS,
        **error_mitigation_options,
    }

    if has_observables:
        program_layers.append(layers.EstimatorLayer(**mitigation_flags))
    else:
        program_layers.append(layers.DistributionMitigationLayer(**mitigation_flags))

    program_layers.append(layers.DeviceLayer(device_id=device_id, options=options, auto_quality=auto_quality))

    return HybridProgram(layers=program_layers), device_credentials
