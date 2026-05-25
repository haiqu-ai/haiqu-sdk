from typing import Dict, Union, Optional
from dataclasses import dataclass


@dataclass
class SolverResult:
    """Result object from the solve() function.

    All bitstrings use Qiskit convention (little-endian): rightmost bit = qubit 0.

    Attributes:
        raw_counts (Dict[str, Union[int, float]]): Raw measurement counts from quantum circuit.
            Keys are bitstrings in Qiskit convention.
        raw_costs (Dict[str, float]): Cost values for raw bitstrings.
            Keys are bitstrings in Qiskit convention.
        processed_counts (Dict[str, Union[int, float]]): Post-processed measurement counts.
            Keys are bitstrings in Qiskit convention.
        processed_costs (Dict[str, float]): Cost values for post-processed bitstrings.
            Keys are bitstrings in Qiskit convention.
        best_solution (str): Best bitstring found (Qiskit convention).
        best_cost (float): Cost of the best solution.
        expectation_value (float): Standard expectation value.
        cvar_expectation (float): Conditional Value at Risk expectation value.
        cvar_alpha (float): Alpha parameter used for CVaR calculation.
        metadata (Dict[str, str]): Metadata including lr_qaoa_circuit_id, compressed_circuit_id, job_id.
        compression_quality (Optional[float]): Compression quality if compression was applied, None otherwise.
        compression_info (Optional[Dict[str, Union[str, int, None]]]): Compression parameters if compression
            was applied, None otherwise.
    """

    raw_counts: Dict[str, Union[int, float]]
    raw_costs: Dict[str, float]
    processed_counts: Dict[str, Union[int, float]]
    processed_costs: Dict[str, float]
    best_solution: str
    best_cost: float
    expectation_value: float
    cvar_expectation: float
    cvar_alpha: float
    metadata: Dict[str, str]
    compression_quality: Optional[float] = None
    compression_info: Optional[Dict[str, Union[str, int, None]]] = None

    def __repr__(self) -> str:
        compression_str = f", compression_quality={self.compression_quality:.6f}" if self.compression_quality is not None else ""
        return (
            f"SolverResult(\n"
            f"  best_solution='{self.best_solution}',\n"
            f"  best_cost={self.best_cost:.6f},\n"
            f"  expectation_value={self.expectation_value:.6f},\n"
            f"  cvar_expectation={self.cvar_expectation:.6f} (alpha={self.cvar_alpha}),\n"
            f"  num_unique_costs={len(self.processed_costs)}{compression_str}\n"
            f")"
        )
