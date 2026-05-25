"""
Custom metrics for SDK examples:

* meyer_wallach
* negativity
* von_neumann_entropy
"""

import numpy as np
from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Statevector
from qiskit.quantum_info import entropy as v_n_entropy
from qiskit.quantum_info import partial_trace
from qiskit_aer import AerSimulator

backend = AerSimulator(method="statevector")


def meyer_wallach(circuit):
    psi = Statevector.from_instruction(circuit)
    n_qubits = circuit.num_qubits

    entanglement_sum = 0
    for k in range(n_qubits):
        indices = [idx for idx in range(n_qubits) if idx is not k]
        rho_k_sq = reduced_density_matrix_qiskit(psi, indices) @ reduced_density_matrix_qiskit(psi, indices)
        entanglement_sum += np.trace(rho_k_sq)

    Q = 2 * (1 - (1 / n_qubits) * entanglement_sum)

    return np.absolute(Q)


def reduced_density_matrix_qiskit(psi, sites):
    psi = Statevector(psi)
    return partial_trace(psi, sites).data


def execute(circuit, shots: int = 1024):
    circuit.measure_all()
    circuit_t = transpile(circuit, backend)
    return backend.run(circuit_t, shots=shots)


def negativity(circuit):
    result = execute(circuit).result()
    final_state = result.get_statevector(circuit)

    # Obtain the density matrix of the final state
    density_matrix = DensityMatrix(final_state)
    partial_transpose_matrix = density_matrix.partial_transpose([1])

    # Compute the negative eigenvalues
    negative_eigenvalues = np.linalg.eigvalsh(partial_transpose_matrix.data)

    # Compute the negativity
    negativity_value = 0.0
    for eigenvalue in negative_eigenvalues:
        negativity_value += abs(eigenvalue)
    negativity_value /= 2.0
    return negativity_value


def von_neumann_entropy(circuit):
    result = execute(circuit).result()
    final_state = result.get_statevector(circuit)

    # Convert the final statevector to a DensityMatrix object
    density_matrix = DensityMatrix(final_state)

    # Calculate the von Neumann entropy
    entropy = v_n_entropy(density_matrix)
    return entropy
