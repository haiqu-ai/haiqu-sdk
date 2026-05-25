from __future__ import annotations
from copy import deepcopy
from typing import Tuple
import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.problems import QuadraticObjective
from qiskit_optimization.translators import from_docplex_mp, from_ising


class QUBO:
    """A class representing a Quadratic Unconstrained Binary Optimization (QUBO) problem.

    This class provides a unified interface for QUBO problems, supporting construction
    from multiple formats and conversion to Qiskit-compatible representations.

    A QUBO problem can be initialized in multiple ways:
    1. From a Docplex model / CPLEX file
    2. From an Ising-like Hamiltonian (SparsePauliOp)
    3. From a Qiskit QuadraticProgram.
    """

    def __init__(self):
        raise RuntimeError(
            "Direct initialization is not allowed. "
            "Please use one of the class constructors: "
            "QUBO.from_quadratic_program(), QUBO.from_file(), "
            "QUBO.from_docplex(), or QUBO.from_hamiltonian()."
        )

    @staticmethod
    def _normalize_diagonal_terms(qp: QuadraticProgram) -> None:
        """Normalize diagonal quadratic terms to linear terms for binary variables.

        For binary variables where x ∈ {0,1}, we have x² = x. Therefore, diagonal
        quadratic terms Q[i,i] should be moved to linear coefficients for consistency.

        This normalization is essential because some conversion methods (e.g., Qiskit's
        from_ising()) may store linear coefficients as diagonal quadratic terms.

        Args:
            qp: QuadraticProgram to normalize (modified in-place)

        Note:
            - Only normalizes diagonal terms for binary variables
            - Integer and continuous variables are skipped (x² ≠ x for these types)
            - Diagonal terms are added to existing linear coefficients
            - Diagonal quadratic entries are set to 0 after normalization
        """
        from qiskit_optimization.problems import Variable

        quad_dict = qp.objective.quadratic.to_dict()
        diagonal_terms = {i: coeff for (i, j), coeff in quad_dict.items() if i == j}

        if diagonal_terms:
            # Get variable list to check types
            variables = list(qp.variables)

            # Only normalize diagonal terms for binary variables
            for i, coeff in diagonal_terms.items():
                var = variables[i]
                # Only normalize if variable is binary (x² = x)
                # Skip integer and continuous variables (x² ≠ x)
                if var.vartype == Variable.Type.BINARY:
                    # Add diagonal term to linear coefficient
                    qp.objective.linear[i] += coeff
                    # Remove diagonal from quadratic
                    qp.objective.quadratic[(i, i)] = 0

    @classmethod
    def from_quadratic_program(cls, qp: QuadraticProgram) -> "QUBO":
        """Create a QUBO from a Qiskit QuadraticProgram.

        Note on Qiskit's Quadratic Coefficient Storage:
            Qiskit's QuadraticExpression internally stores quadratic coefficients in
            **upper-triangle format only** (i.e., only :math:`Q_{ij}` for :math:`i < j`).

            The objective function form is:

            .. math::

                f(x) = c + \\sum_i a_i x_i + \\sum_i \\sum_{j>i} Q_{ij} x_i x_j

            **Important Details:**

            - When you call ``qp.objective.quadratic.to_dict()``, it returns only the upper triangle
            - Each stored :math:`Q_{ij}` represents the **full coefficient** of :math:`x_i x_j` (not doubled)
            - When setting coefficients via ``qp.objective.quadratic[(i,j)]``, indices are normalized
              to ``(min(i,j), max(i,j))``, so setting Q[1,0] overwrites Q[0,1]
            - When passing a full symmetric matrix to ``minimize(quadratic=[[...]])``, Qiskit **sums**
              the symmetric entries (e.g., Q[0,1]=1 and Q[1,0]=1 → stored as Q[0,1]=2)

            All QUBO methods in this class properly handle Qiskit's upper-triangle representation.
        """
        # Normalize diagonal quadratic terms before conversion
        # For binary variables: x_i^2 = x_i, so diagonal terms should be linear
        qp = deepcopy(qp)
        cls._normalize_diagonal_terms(qp)

        to_qubo = QuadraticProgramToQubo()
        qubo = to_qubo.convert(qp)

        # Sanity checks
        if qubo.get_num_vars() == 0:
            raise ValueError("QUBO must contain at least one binary variable.")
        if qubo.get_num_linear_constraints() or qubo.get_num_quadratic_constraints():
            raise ValueError("QUBO must be unconstrained after conversion.")

        obj = qubo.objective
        instance = object.__new__(cls)
        instance._qp = qubo
        instance._var_names = tuple(qubo.variables_index.keys())
        instance._offset = float(obj.constant) if isinstance(obj, QuadraticObjective) else 0.0
        return instance

    @classmethod
    def from_file(cls, path: str) -> "QUBO":
        """Load a problem from a CPLEX/LP file and convert it to QUBO form."""
        qp = QuadraticProgram()
        ext = path.lower().rsplit(".", 1)[-1]
        if ext == "lp":
            qp.read_from_lp_file(path)
        else:
            raise ValueError(f"Unsupported file extension '.{ext}'. Please provide a .lp file.")
        return cls.from_quadratic_program(qp)

    @classmethod
    def from_docplex(cls, docplex_model) -> "QUBO":
        """Create from a DOcplex model (docplex.mp.model.Model)."""
        qp = from_docplex_mp(docplex_model)
        return cls.from_quadratic_program(qp)

    @classmethod
    def from_hamiltonian(cls, H: SparsePauliOp, offset: float = 0.0) -> "QUBO":
        """Create QUBO from an Ising Hamiltonian represented as a Pauli operator.

        This method converts an Ising model Hamiltonian (with spin variables sᵢ ∈ {-1, +1})
        to QUBO formulation (with binary variables xᵢ ∈ {0, 1}) using the mapping sᵢ = 1 - 2·xᵢ.

        **Input Format:**
            The Hamiltonian is given as a SparsePauliOp containing Pauli Z operators:

            - Single Z terms (e.g., 'Z', 'IZI') represent local fields hᵢ
            - Products of Zs (e.g., 'ZZ', 'IZZI') represent couplings Jᵢⱼ
            - Pauli X or Y operators are not supported (pure Ising model)

        **Conversion Formula:**
            Ising Hamiltonian: H = Σ hᵢ·sᵢ + Σ Jᵢⱼ·sᵢ·sⱼ + offset  (sᵢ ∈ {-1, +1})

            The conversion uses the mapping: sᵢ = 1 - 2·xᵢ where xᵢ ∈ {0, 1}

            This transforms to QUBO: f(x) = c + Σ aᵢ·xᵢ + Σ Qᵢⱼ·xᵢ·xⱼ

            **Detailed Conversion for Each Term Type:**

            1. **Local field term** hᵢ·sᵢ (where sᵢ ∈ {-1, +1}):

               Substitute sᵢ = 1 - 2·xᵢ:
                   = hᵢ·(1 - 2·xᵢ)
                   = hᵢ - 2·hᵢ·xᵢ

               Contributes:
                   - Constant: +hᵢ
                   - Linear xᵢ: -2·hᵢ

            2. **Coupling term** Jᵢⱼ·sᵢ·sⱼ:

               Substitute sᵢ = 1-2·xᵢ and sⱼ = 1-2·xⱼ:
                   = Jᵢⱼ·(1 - 2·xᵢ)·(1 - 2·xⱼ)
                   = Jᵢⱼ·[1 - 2·xᵢ - 2·xⱼ + 4·xᵢ·xⱼ]

               Contributes:
                   - Constant: +Jᵢⱼ
                   - Linear xᵢ: -2·Jᵢⱼ
                   - Linear xⱼ: -2·Jᵢⱼ
                   - Quadratic xᵢ·xⱼ: +4·Jᵢⱼ

            **Important:** When multiple Pauli terms are present, their contributions are **summed**.
            For example, if both hᵢ·Zᵢ and Jᵢⱼ·Zᵢ·Zⱼ affect variable xᵢ, the linear coefficients
            add: aᵢ = -2·hᵢ + (-2·Jᵢⱼ) = -2·(hᵢ + Jᵢⱼ).

            **Note on Normalization:** Qiskit's ``from_ising()`` stores linear terms as diagonal
            entries in the quadratic matrix (e.g., -2·hᵢ becomes Q[i,i] = -2·hᵢ). These diagonal
            terms are automatically normalized to linear coefficients by ``from_quadratic_program()``,
            since for binary variables xᵢ² = xᵢ.

        **Implementation:**
            Uses Qiskit's ``from_ising()`` function to perform the conversion, which handles
            the Pauli operator algebra and coefficient transformations automatically. The
            resulting QuadraticProgram is then normalized via ``from_quadratic_program()``.

        Args:
            H: Ising Hamiltonian as a SparsePauliOp (must contain only Z operators)
            offset: Additional constant offset to add to the Hamiltonian. Defaults to 0.0.

        Returns:
            QUBO instance representing the same optimization problem

        Raises:
            TypeError: If H is not a SparsePauliOp
            QiskitOptimizationError: If H contains Pauli X or Y operators
            QiskitOptimizationError: If any Pauli term acts on more than 2 qubits (only pairwise interactions supported)

        Example:
            >>> from qiskit.quantum_info import SparsePauliOp
            >>>
            >>> # Create simple Ising Hamiltonian: H = 1.0·Z₀·Z₁ (coupling only)
            >>> # This represents interaction between spins: J₀₁·s₀·s₁ with J₀₁ = 1.0
            >>> H = SparsePauliOp.from_list([('ZZ', 1.0)])
            >>> qubo = QUBO.from_hamiltonian(H, offset=0.0)
            >>>
            >>> # The conversion applies: J₀₁·s₀·s₁ = J₀₁·(1-2x₀)·(1-2x₁)
            >>> # Expanding: J₀₁ - 2·J₀₁·x₀ - 2·J₀₁·x₁ + 4·J₀₁·x₀·x₁
            >>> # With J₀₁ = 1.0:
            >>> #   Constant: +1.0
            >>> #   Linear x₀: -2.0 (normalized from diagonal Q[0,0])
            >>> #   Linear x₁: -2.0 (normalized from diagonal Q[1,1])
            >>> #   Quadratic x₀·x₁: +4.0
            >>>
            >>> print(qubo._qp.objective.constant)  # 1.0
            >>> print(qubo._qp.objective.linear.to_dict())  # {0: -2.0, 1: -2.0}
            >>> print(qubo._qp.objective.quadratic.to_dict())  # {(0,1): 4.0}
        """
        if not isinstance(H, SparsePauliOp):
            raise TypeError("Input must be a qiskit.quantum_info.SparsePauliOp.")
        qp = from_ising(H, offset=offset)

        # Normalization of diagonal terms is handled by from_quadratic_program()
        return cls.from_quadratic_program(qp)

    def to_hamiltonian(self) -> Tuple[SparsePauliOp, float]:
        """Convert QUBO to an Ising Hamiltonian represented as Pauli operators.

        This method performs the **inverse transformation** of ``from_hamiltonian()``,
        converting from QUBO formulation (binary variables xᵢ ∈ {0, 1}) back to Ising
        model (spin variables sᵢ ∈ {-1, +1}) using the inverse mapping xᵢ = (1 - sᵢ)/2.

        **Output Format:**
            Returns a tuple ``(H, offset)`` where:

            - ``H``: SparsePauliOp containing the Ising Hamiltonian as Pauli Z operators
            - ``offset``: Float constant offset term

        **Conversion Formula (QUBO → Ising):**
            QUBO objective: f(x) = c + Σ aᵢ·xᵢ + Σ Qᵢⱼ·xᵢ·xⱼ  (xᵢ ∈ {0, 1})

            Using inverse substitution xᵢ = (1 - sᵢ)/2, this becomes:

            Ising Hamiltonian expectation: ⟨H⟩ = Σ hᵢ·sᵢ + Σ Jᵢⱼ·sᵢ·sⱼ + offset  (sᵢ ∈ {-1, +1})

            Where the coefficients are derived by reversing the Ising→QUBO transformation.

        **Implementation:**
            Uses Qiskit's ``QuadraticProgram.to_ising()`` method to perform the conversion,
            which handles the variable substitution and Pauli operator construction automatically.

        Returns:
            Tuple[SparsePauliOp, float]: A tuple containing:
                - SparsePauliOp: The Ising Hamiltonian with Pauli Z operators
                - float: The constant offset term

        Example:
            >>> # Create QUBO with linear and quadratic terms
            >>> from qiskit_optimization import QuadraticProgram
            >>> qp = QuadraticProgram()
            >>> qp.binary_var('x0')
            >>> qp.binary_var('x1')
            >>> qp.minimize(linear={'x0': 0.5}, quadratic={('x0', 'x1'): 1.0})
            >>> qubo = QUBO.from_quadratic_program(qp)
            >>> H, offset = qubo.to_hamiltonian()
            >>> # H is a SparsePauliOp like: 0.5*Z_0 + 1.0*Z_0*Z_1
            >>> # offset is the constant term
            >>> # Can now use with Qiskit quantum algorithms:
            >>> from qiskit_algorithms import QAOA
            >>> qaoa = QAOA(sampler=sampler, optimizer=optimizer)
            >>> result = qaoa.compute_minimum_eigenvalue(H)

        See Also:
            - ``from_hamiltonian()``: Inverse operation (Ising → QUBO)
        """
        op, offset = self._qp.to_ising()
        if not isinstance(op, SparsePauliOp):
            op = SparsePauliOp.from_list(op.to_list())
        return op, float(offset)

    def to_file(self, path: str) -> str:
        """Export the QUBO as a CPLEX LP file."""
        if not path.lower().endswith(".lp"):
            raise ValueError("Output path must have a .lp extension.")
        self._qp.write_to_lp_file(path)
        return path

    def to_lp_string(self) -> str:
        """Serialize QUBO to LP file format string.

        Returns:
            str: LP file content as string
        """
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".lp", mode="w", delete=False) as f:
            temp_path = f.name

        try:
            self.to_file(temp_path)
            with open(temp_path, "r") as f:
                lp_content = f.read()
            return lp_content
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @classmethod
    def from_lp_string(cls, lp_content: str) -> "QUBO":
        """Deserialize QUBO from LP file format string.

        Args:
            lp_content: LP file content as string

        Returns:
            QUBO instance
        """
        import tempfile
        import os

        with tempfile.NamedTemporaryFile(suffix=".lp", mode="w", delete=False) as f:
            f.write(lp_content)
            temp_path = f.name

        try:
            problem = cls.from_file(temp_path)
            return problem
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    def cost(self, bitstring: str) -> float:
        """Compute the QUBO objective for a given bitstring.

        Uses the formula: :math:`f(x) = c + a^T x + \\sum_{i<j} Q_{ij} x_i x_j`
        where :math:`x_i \\in \\{0, 1\\}` are binary variables.

        Note on Qiskit's Quadratic Coefficient Storage:
            Qiskit's QuadraticExpression stores coefficients in **upper-triangle format only**:

            .. math::

                f(x) = c + \\sum_i a_i x_i + \\sum_i \\sum_{j>i} Q_{ij} x_i x_j

            - Only :math:`Q_{ij}` for :math:`i < j` are stored (via ``to_dict()``)
            - Each stored :math:`Q_{ij}` is the **full coefficient** (not doubled)
            - Example: If ``to_dict()`` returns ``{(0,1): 0.5}``, the contribution is ``0.5 * x_0 * x_1``

            **Important:** When you pass a symmetric matrix to ``minimize(quadratic=[[...]])``,
            Qiskit sums symmetric entries. For instance, ``[[0,1],[1,0]]`` stores Q[0,1]=2.

        Args:
            bitstring: Bitstring in Qiskit convention (little-endian, rightmost bit = qubit 0).

        Returns:
            float: The objective value for the given bitstring.

        Note:
            Bitstrings use Qiskit convention: rightmost bit = qubit 0.
            Example: "101" means x0=1, x1=0, x2=1 (for var_names=['x0','x1','x2']).
        """
        # Reverse bitstring from Qiskit convention (little-endian) to problem order (big-endian)
        bitstring = bitstring[::-1]
        x = self._normalize_bitstring(bitstring)

        # Get coefficients from internal QuadraticProgram
        c = self._offset
        lin_dict = self._qp.objective.linear.to_dict()
        quad_dict = self._qp.objective.quadratic.to_dict()

        # Constant term
        E = c

        # Linear term: Σ a_i * x_i
        for i, coeff in lin_dict.items():
            E += float(coeff) * x[i]

        # Quadratic term: Σ Q_ij * x_i * x_j
        for (i, j), coeff in quad_dict.items():
            E += float(coeff) * x[i] * x[j]

        return float(E)

    def _normalize_bitstring(self, bitstring: str) -> np.ndarray:
        n = len(self._var_names)
        if isinstance(bitstring, str):
            if len(bitstring) != n or any(ch not in "01" for ch in bitstring):
                raise ValueError(f"Bitstring must have length {n} and only contain '0'/'1'.")
            return np.array([int(ch) for ch in bitstring], dtype=float)

    @property
    def var_names(self) -> Tuple[str, ...]:
        return self._var_names

    @property
    def num_vars(self) -> int:
        return len(self._var_names)
