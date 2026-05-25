"""
Inverse QFT circuit that operates on counting qubits.
"""

import math

from qiskit import QuantumCircuit, QuantumRegister


def inv_qft_gate(input_size):
    """
    Creates inverse QFT circuit that operates on counting qubits

    param input_size: the number of counting qubits
    return: qiskit.QuantumCircuit object with IQFT circuit
    """
    qr = QuantumRegister(input_size)
    qc = QuantumCircuit(qr, name="inv_qft")

    # Generate multiple groups of diminishing angle CRZs and H gate
    for i_qubit in reversed(range(0, input_size)):
        # start laying out gates from highest order qubit (the hidx)
        hidx = input_size - i_qubit - 1

        # precede with an H gate (applied to all qubits)
        qc.h(qr[hidx])

        # if not the highest order qubit, add multiple controlled RZs of decreasing angle
        if hidx < input_size - 1:
            num_crzs = i_qubit
            for j in reversed(range(0, num_crzs)):
                divisor = 2 ** (num_crzs - j)
                qc.crz(-math.pi / divisor, qr[hidx], qr[input_size - j - 1])

    return qc
