"""
Quantum Amplitude Estimation

Quantum Amplitude Estimation (QAE) is an algorithm that provides a quadratic
speedup over classical Monte Carlo methods for estimating the expectation value
of a random variable defined by a quantum state.
"""

import copy

import numpy as np
import qiskit

from haiqu.sdk.library.inv_qft import inv_qft_gate


def Ctrl_Q(num_state_qubits, A_circ):
    """
    Construct the grover-like operator and a controlled version of it

    param num_state_qubits: the number of state qubits
    param A_circ: A operator circuit
    return: qiskit.QuantumCirciut object with Quantum Operator circuit
    """

    # index n is the objective qubit, and indexes 0 through n-1 are state qubits
    qc = qiskit.QuantumCircuit(num_state_qubits + 1, name="Q")

    temp_A = copy.copy(A_circ)
    A_gate = temp_A.to_gate()
    A_gate_inv = temp_A.inverse().to_gate()

    # Each cycle in Q applies in order: -S_chi, A_circ_inverse, S_0, A_circ
    # -S_chi
    qc.x(num_state_qubits)
    qc.z(num_state_qubits)
    qc.x(num_state_qubits)

    # A_circ_inverse
    qc.append(A_gate_inv, [i for i in range(num_state_qubits + 1)])

    # S_0
    for i in range(num_state_qubits + 1):
        qc.x(i)
    qc.h(num_state_qubits)

    qc.mcx([x for x in range(num_state_qubits)], num_state_qubits)

    qc.h(num_state_qubits)
    for i in range(num_state_qubits + 1):
        qc.x(i)

    # A_circ
    qc.append(A_gate, [i for i in range(num_state_qubits + 1)])

    # Create a gate out of the Q operator
    qc.to_gate(label="Q")

    # and also a controlled version of it
    Ctrl_Q_ = qc.control(1)

    # and return both
    return Ctrl_Q_, qc


def A_gen(num_state_qubits, a, psi_zero=None, psi_one=None):
    """
    Construct A operator that takes |0>_{n+1} to sqrt(1-a) |psi_0>|0> + sqrt(a) |psi_1>|1>

    param num_state_qubits: the number of state qubits
    param a: amplitude value
    param psi_zero: defines one of the two states AE operates on
    param psi_one: defines one of the two states AE operates on
    return: qiskit.QuantumCircuit object of operator A
    """

    if psi_zero is None:
        psi_zero = "0" * num_state_qubits
    if psi_one is None:
        psi_one = "1" * num_state_qubits

    theta = 2 * np.arcsin(np.sqrt(a))
    # Let the objective be qubit index n; state is on qubits 0 through n-1
    qc_A = qiskit.QuantumCircuit(num_state_qubits + 1, name="A")

    # takes state to |0>_{n} (sqrt(1-a) |0> + sqrt(a) |1>)
    qc_A.ry(theta, num_state_qubits)

    # takes state to sqrt(1-a) |psi_0>|0> + sqrt(a) |0>_{n}|1>
    qc_A.x(num_state_qubits)
    for i in range(num_state_qubits):
        if psi_zero[i] == "1":
            qc_A.cx(num_state_qubits, i)
    qc_A.x(num_state_qubits)

    # takes state to sqrt(1-a) |psi_0>|0> + sqrt(a) |psi_1>|1>
    for i in range(num_state_qubits):
        if psi_one[i] == "1":
            qc_A.cx(num_state_qubits, i)

    return qc_A


def AmplitudeEstimation(name, n_qubits, num_state_qubits, seed=None):
    """
    Creates a circuit out of Amplitude Generator and Quantum Operator circuits.

    param n_qubits: total number of qubits
    param num_state_qubits: the number of state qubits (qubits that Amplitude Generator operates on)
    param seed: random seed to set
    return: qiskit.QuantumCircuit object of AE circuit
    """
    qc = qiskit.QuantumCircuit(n_qubits)
    qc.name = name
    num_counting_qubits = n_qubits - num_state_qubits - 1

    np.random.seed(seed)
    amplitude = np.random.sample()

    # create the Amplitude Generator circuit
    A = A_gen(num_state_qubits, amplitude, psi_zero=None, psi_one=None)

    # create the Quantum Operator circuit and a controlled version of it
    cQ, Q = Ctrl_Q(num_state_qubits, A)

    # Prepare state from A, and counting qubits with H transform
    qc.append(A, [num_counting_qubits + i for i in range(num_state_qubits + 1)])
    for i in range(num_counting_qubits):
        qc.h(i)

    repeat = 1
    for j in reversed(range(num_counting_qubits)):
        for _ in range(repeat):
            qc.append(cQ, [j] + [num_counting_qubits + x for x in range(num_state_qubits + 1)])
        repeat *= 2

    # inverse quantum Fourier transform only on counting qubits
    qc.append(inv_qft_gate(num_counting_qubits), range(num_counting_qubits))

    return qc
