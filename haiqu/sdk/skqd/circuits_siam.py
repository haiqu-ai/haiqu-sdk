"""
Krylov subspace circuit generation for the SIAM model (momentum basis).
"""

import numpy as np
import scipy.linalg
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XGate, CPhaseGate, XXPlusYYGate

try:
    from ffsim.qiskit import OrbitalRotationJW
except ImportError:
    OrbitalRotationJW = None

_FFSIM_IMPORT_ERROR = (
    "The 'ffsim' package is required for SKQD circuit generation but is not installed.\n"
    "Install it with:  pip install haiqu-sdk[skqd]\n"
    "Note: ffsim requires a Rust toolchain (Cargo). "
    "See https://rustup.rs/ for installation instructions."
)


def _prepare_siam_initial_state(qubits: QuantumRegister, norb: int, nocc: int) -> QuantumCircuit:
    """Prepare initial state for SIAM in momentum basis.

    Fills the lowest-energy momentum orbitals using X gates, then applies
    XXPlusYY rotations to spread the state across neighboring orbitals.

    Args:
        qubits: Quantum register with 2*norb qubits.
        norb: Number of spatial orbitals.
        nocc: Number of occupied orbitals per spin (typically norb // 2).

    Returns:
        Quantum circuit that prepares the state.
    """
    circuit = QuantumCircuit(qubits)
    x_gate = XGate()
    rot = XXPlusYYGate(0.5 * np.pi, -0.5 * np.pi)

    for i in range(nocc):
        circuit.append(x_gate, [qubits[i]])
        circuit.append(x_gate, [qubits[norb + i]])

    # Spreading rotations outward from the Fermi level.
    # Each layer i touches indices from (nocc - i - 1) to (nocc + i).
    # Constraints: lower index >= 0  and  upper index + 1 < norb.
    n_layers = min(3, nocc, norb - nocc)
    j = None
    for i in range(n_layers):
        for j in range(nocc - i - 1, nocc + i, 2):
            circuit.append(rot, [qubits[j], qubits[j + 1]])
            circuit.append(rot, [qubits[norb + j], qubits[norb + j + 1]])

    # One additional rotation beyond the last layer, if it fits.
    if j is not None and j + 2 < norb:
        circuit.append(rot, [qubits[j + 1], qubits[j + 2]])
        circuit.append(rot, [qubits[norb + j + 1], qubits[norb + j + 2]])

    return circuit


def _siam_trotter_step(
    qubits: QuantumRegister,
    dt: float,
    h1e: np.ndarray,
    h2e: np.ndarray,
    impurity_index: int,
    norb: int,
) -> QuantumCircuit:
    """Second-order Trotter step for SIAM in momentum basis.

    In momentum basis, the two-body interaction is localized to the impurity
    orbital only, so only a single CPhase gate is needed (vs all orbitals
    for Hubbard).

    Args:
        qubits: Quantum register with 2*norb qubits.
        dt: Time step size.
        h1e: One-body Hamiltonian tensor in momentum basis.
        h2e: Two-body tensor in momentum basis.
        impurity_index: Index of the impurity orbital in momentum ordering.
        norb: Number of spatial orbitals.

    Returns:
        Quantum circuit implementing the Trotter step.
    """
    circuit = QuantumCircuit(qubits)

    one_body_evolution = scipy.linalg.expm(-1j * dt * h1e)

    onsite = h2e[impurity_index, impurity_index, impurity_index, impurity_index]

    # Two-body evolution for half the time (impurity only)
    circuit.append(
        CPhaseGate(-0.5 * dt * onsite),
        [qubits[impurity_index], qubits[norb + impurity_index]],
    )

    # One-body evolution for the full time
    circuit.append(OrbitalRotationJW(norb, one_body_evolution), qubits)

    # Two-body evolution for half the time (impurity only)
    circuit.append(
        CPhaseGate(-0.5 * dt * onsite),
        [qubits[impurity_index], qubits[norb + impurity_index]],
    )

    return circuit


def build_siam_momentum_basis_krylov_circuits(
    norb: int,
    krylov_dim: int,
    dt: float,
    h1e: np.ndarray,
    h2e: np.ndarray,
    impurity_index: int,
) -> list[QuantumCircuit]:
    """Build Krylov subspace circuits for SIAM in momentum basis.

    The ground state of the Anderson impurity model is significantly sparser
    in momentum basis -- fewer determinants carry significant weight -- making
    SQD sampling far more effective than in site basis.

    The Trotter step applies a single CPhase on the impurity orbital only
    (since U is localized there in momentum basis).

    Args:
        norb: Number of spatial orbitals.
        krylov_dim: Dimension of the Krylov subspace
            (number of time-evolution steps + 1).
        dt: Time step for Trotter evolution.
        h1e: One-body Hamiltonian tensor in momentum basis, shape (norb, norb).
        h2e: Two-body Hamiltonian tensor in momentum basis,
            shape (norb, norb, norb, norb).
        impurity_index: Index of the impurity orbital in momentum ordering.

    Returns:
        List of quantum circuits representing the Krylov subspace.
    """
    if OrbitalRotationJW is None:
        raise ImportError(_FFSIM_IMPORT_ERROR)

    nocc = norb // 2
    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)

    # Prepare initial state
    circuit.compose(_prepare_siam_initial_state(qubits, norb, nocc), inplace=True)

    # Snapshot initial state with measurements
    circuit.measure_all()
    circuits = [circuit.copy()]

    # Pre-compute single Trotter step circuit
    trotter_circuit = _siam_trotter_step(
        qubits,
        dt,
        h1e,
        h2e,
        impurity_index,
        norb,
    )

    # Apply time evolution iteratively
    for _ in range(krylov_dim - 1):
        # Remove measurements before appending next Trotter step
        circuit.remove_final_measurements()
        circuit.compose(trotter_circuit, inplace=True)
        circuit.measure_all()
        circuits.append(circuit.copy())

    return circuits
