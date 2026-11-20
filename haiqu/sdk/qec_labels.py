"""
Haiqu SDK. QEC box labels.

The QEC compiler wraps each logical operation of a compiled circuit in a labeled box. This module
builds those labels, so that they do not have to be typed by hand when selecting boxes in the
``toggles`` argument of ``haiqu.qec_run``, either in ``"noisy_box_labels"`` or in
``"qubit_gate_overrides"``.

Every label is the shared prefix, a fixed operation term, and, for a parameterized operation, a
single space and one ``key=value`` tail::

    __qec_compiler__:ec_cycle
    __qec_compiler__:logical_state_preparation state=0
    __qec_compiler__:logical_gate name=x
    __qec_compiler__:data_measurement basis=Z
    __qec_compiler__:logical_reset state=0
    __qec_compiler__:logical_measurement observable=Z
"""

QEC_COMPILER_BOX_LABEL_PREFIX = "__qec_compiler__:"
"""Prefix shared by every label the compiler puts on a box."""


def ec_cycle_box_label() -> str:
    """Label on the box wrapping each error-correction cycle."""
    return f"{QEC_COMPILER_BOX_LABEL_PREFIX}ec_cycle"


def logical_state_preparation_box_label(state: str) -> str:
    """Label on the box wrapping a logical state preparation (e.g. state='0')."""
    return f"{QEC_COMPILER_BOX_LABEL_PREFIX}logical_state_preparation {state=:}"


def logical_gate_box_label(gate_name: str) -> str:
    """Label on the box wrapping a logical gate (e.g. gate_name='x')."""
    return f"{QEC_COMPILER_BOX_LABEL_PREFIX}logical_gate name={gate_name}"


def data_measurement_box_label(basis: str) -> str:
    """Label on the box wrapping a data measurement (e.g. basis='Z')."""
    return f"{QEC_COMPILER_BOX_LABEL_PREFIX}data_measurement {basis=:}"


def logical_reset_box_label(state: str) -> str:
    """Label on the box wrapping a logical reset (e.g. state='0')."""
    return f"{QEC_COMPILER_BOX_LABEL_PREFIX}logical_reset {state=:}"


def logical_measurement_box_label(observable: str) -> str:
    """Label on the box wrapping a logical measurement (e.g. observable='Z')."""
    return f"{QEC_COMPILER_BOX_LABEL_PREFIX}logical_measurement {observable=:}"
