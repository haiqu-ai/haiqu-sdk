from typing import Optional, Tuple

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from qiskit_addon_utils.slicing import slice_by_gate_types
from qiskit_addon_obp.utils.simplify import OperatorBudget
from qiskit_addon_obp.utils.truncating import setup_budget
from qiskit_addon_obp import backpropagate
from qiskit_addon_utils.slicing import combine_slices


def backpropagation(
    circuit: QuantumCircuit,
    observable: SparsePauliOp,
    max_qwc_groups: Optional[int] = None,
    max_error_total: Optional[float] = None,
    max_error_per_slice: Optional[float] = None,
) -> Tuple[QuantumCircuit, SparsePauliOp]:
    """Backpropagate an observable through a quantum circuit using operator budget.

    Args:
        circuit (QuantumCircuit): The quantum circuit to backpropagate through.
        observable (SparsePauliOp): The observable to be backpropagated.
        max_qwc_groups (Optional[int]): Maximum number of QWC groups.
        max_error_total (Optional[float]): Maximum total error allowed.
        max_error_per_slice (Optional[float]): Maximum error allowed per slice.

    Returns:
        Tuple[QuantumCircuit, SparsePauliOp]: The backpropagated circuit and observable.
    """
    op_budget = OperatorBudget(max_qwc_groups=max_qwc_groups)
    if max_error_total is not None or max_error_per_slice is not None:
        truncation_error_budget = setup_budget(max_error_total=max_error_total, max_error_per_slice=max_error_per_slice)
    else:
        truncation_error_budget = None

    slices = slice_by_gate_types(circuit)

    try:
        backpropagated_observable, remaining_slices, _ = backpropagate(
            observable,
            slices,
            operator_budget=op_budget,
            truncation_error_budget=truncation_error_budget,
        )
    except MemoryError as e:
        raise MemoryError(
            f"MemoryError {str(e)} was raised during backpropagation. "
            "This can often happen if the circuit contains ambiguous operations. "
            "Try transpiling the circuit to a common universal gateset (e.g. RX, RZ, RZZ)"
        )

    # Recombine the slices remaining after backpropagation
    if len(remaining_slices) == 0:
        bp_circuit = QuantumCircuit.copy_empty_like(circuit)
    else:
        bp_circuit = combine_slices(remaining_slices)

    return bp_circuit, backpropagated_observable
