"""
Krylov subspace circuit generation for the Hubbard model (site basis).
"""

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import XGate, CPhaseGate

try:
    from ffsim.qiskit import OrbitalRotationJW, NumOpSumEvolutionJW
except ImportError:
    OrbitalRotationJW = None
    NumOpSumEvolutionJW = None

_FFSIM_IMPORT_ERROR = (
    "The 'ffsim' package is required for SKQD circuit generation but is not installed.\n"
    "Install it with:  pip install haiqu-sdk[skqd]\n"
    "Note: ffsim requires a Rust toolchain (Cargo). "
    "See https://rustup.rs/ for installation instructions."
)


def _prepare_hubbard_initial_state(qubits: QuantumRegister, norb: int) -> QuantumCircuit:
    """Prepare antiferromagnetic initial state |up down up down ...>.

    Under Jordan-Wigner mapping with 2*norb qubits, the first `norb` qubits
    are spin-up orbitals and the last `norb` are spin-down orbitals.

    Args:
        qubits: Quantum register with 2*norb qubits.
        norb: Number of spatial orbitals.

    Returns:
        Quantum circuit that prepares the state.
    """
    circuit = QuantumCircuit(qubits)
    x_gate = XGate()
    for i in range(norb // 2):
        circuit.append(x_gate, [qubits[2 * i]])
        circuit.append(x_gate, [qubits[2 * i + norb + 1]])
    return circuit


def _hubbard_trotter_step(
    qubits: QuantumRegister,
    dt: float,
    h1e: np.ndarray,
    h2e: np.ndarray,
    norb: int,
) -> QuantumCircuit:
    """Second-order Trotter step for Hubbard model (site basis).

    exp(-iHt) ~ exp(-iH2 t/2) exp(-iH1 t) exp(-iH2 t/2).

    Site-basis decomposition:
        1. H2 half-step (diagonal in site basis: CPhase gates on all orbitals)
        2. Rotate site -> momentum
        3. H1 full-step (diagonal in momentum basis: NumOpSum)
        4. Rotate momentum -> site
        5. H2 half-step

    Args:
        qubits: Quantum register with 2*norb qubits.
        dt: Time step size.
        h1e: One-body Hamiltonian in site basis.
        h2e: Two-body Hamiltonian in site basis.
        norb: Number of spatial orbitals.

    Returns:
        Quantum circuit implementing the Trotter step.
    """
    circuit = QuantumCircuit(qubits)

    eigenvalues, eigenvectors = np.linalg.eigh(h1e)
    mom_rot = eigenvectors.conj().T  # site -> momentum
    site_rot = eigenvectors  # momentum -> site
    coeffs = np.real(eigenvalues)

    # H2 half-step
    for i in range(norb):
        circuit.append(
            CPhaseGate(-0.5 * dt * h2e[i, i, i, i]),
            [qubits[i], qubits[norb + i]],
        )

    # site -> momentum
    circuit.append(OrbitalRotationJW(norb, mom_rot), qubits)

    # H1 full-step
    circuit.append(NumOpSumEvolutionJW(norb, coeffs, dt), qubits)

    # momentum -> site
    circuit.append(OrbitalRotationJW(norb, site_rot), qubits)

    # H2 half-step
    for i in range(norb):
        circuit.append(
            CPhaseGate(-0.5 * dt * h2e[i, i, i, i]),
            [qubits[i], qubits[norb + i]],
        )

    return circuit


def build_hubbard_site_basis_krylov_circuits(
    norb: int,
    krylov_dim: int,
    dt: float,
    h1e: np.ndarray,
    h2e: np.ndarray,
) -> list[QuantumCircuit]:
    """Build Krylov subspace circuits for the Hubbard model in site basis.

    Generates ``{|psi0>, e^{-iHdt}|psi0>, e^{-2iHdt}|psi0>, ...}`` where
    ``|psi0>`` is the antiferromagnetic Neel state and time evolution uses
    a second-order Trotter decomposition with CPhase gates on all orbitals.

    Args:
        norb: Number of spatial orbitals.
        krylov_dim: Dimension of the Krylov subspace
            (number of time-evolution steps + 1).
        dt: Time step for Trotter evolution.
        h1e: One-body Hamiltonian tensor, shape (norb, norb).
        h2e: Two-body Hamiltonian tensor, shape (norb, norb, norb, norb).

    Returns:
        List of quantum circuits representing the Krylov subspace.
    """
    if OrbitalRotationJW is None:
        raise ImportError(_FFSIM_IMPORT_ERROR)

    qubits = QuantumRegister(2 * norb, name="q")
    circuit = QuantumCircuit(qubits)

    # Prepare initial state
    circuit.compose(_prepare_hubbard_initial_state(qubits, norb), inplace=True)

    circuits = []
    snap = circuit.copy()
    snap.measure_all()
    circuits.append(snap)

    # Pre-compute single Trotter step circuit
    trotter_circuit = _hubbard_trotter_step(qubits, dt, h1e, h2e, norb)

    # Apply time evolution iteratively
    for _ in range(krylov_dim - 1):
        circuit.compose(trotter_circuit, inplace=True)
        snap = circuit.copy()
        snap.measure_all()
        circuits.append(snap)

    return circuits
