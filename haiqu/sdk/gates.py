"""
Haiqu SDK. Custom gates.
"""

from qiskit.circuit import Instruction
from qiskit.circuit.exceptions import CircuitError


class HaiquCircuitGate(Instruction):
    """
    Gate for embedding a circuit stored in the Haiqu cloud.

    If you have access to a circuit stored in Haiqu (for example, from a data-loading job result,
    :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.get_circuit`, or any
    :class:`~haiqu.sdk.schemas.CircuitModel` converted with ``to_gate()``), you can insert it as a
    sub-circuit into a larger quantum algorithm as a :class:`HaiquCircuitGate`.

    .. note::

        This gate is a handle for a circuit stored server-side: it carries only a ``circuit_id``,
        its ``definition`` is ``None``, and it is expanded inside the Haiqu cloud at submission time.
        Obtain instances from data-loading job results, from
        :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.get_circuit` via :meth:`~haiqu.sdk.schemas.CircuitModel.to_gate`,
        or from any :class:`~haiqu.sdk.schemas.CircuitModel` with ``to_gate()``; do not construct circuit IDs manually.

        Client-side composition works as for any ``qiskit.circuit.Instruction``: ``circuit.append(...)``,
        ``circuit.compose(...)``, and ``inverse()`` (returns a :class:`HaiquCircuitdgGate`). Drawing and
        submission via :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.transpile` and
        :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.run` will work for circuits containing a :class:`HaiquCircuitGate`.

        Local expansion does not work — avoid the following on circuits containing this gate:

        * Local simulation (``qiskit_aer``, ``Statevector``): the matrix form is not available client-side.
        * ``qiskit.transpile`` / ``circuit.decompose()`` cannot expand the server-side circuit
          definition. Use :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.transpile` instead.
        * ``circuit.depth()`` and local gate counts: misleading on circuits containing this gate.
          Transpile with :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.transpile` and read
          ``.analytics`` on the returned :class:`~haiqu.sdk.schemas.CircuitModel` (for example
          ``.analytics.depth``, ``.analytics.gates_2q``) or ``.core_metrics()``.

    **Circuit symbol:**

    .. parsed-literal::

             ┌─────────────────────────────────────┐
        q_0: ┤                                     ├
        ...  ┤ Haiqucircuit(circuit_id,num_qubits) ├
        q_N  ┤                                     ├
             └─────────────────────────────────────┘
    """

    def __init__(
        self,
        circuit_id: str,
        num_qubits: int | float,
        *,
        label: str | None = None,
        name: str = "HaiquCircuit",
    ):
        """
        Args:
            circuit_id (str): The ID of the generated quantum circuit.
            num_qubits (int | float): The number of qubits on which the circuit operates.
            label (str | None): An optional label for identifying the instruction.
        """
        if not (isinstance(circuit_id, str) and circuit_id.startswith("circ-")):
            raise ValueError("Invalid circuit ID")

        super().__init__(name, int(num_qubits), 0, [circuit_id, num_qubits], label=label)
        self.circuit_id = circuit_id

    def __repr__(self):
        return f"{self.__class__.__name__}({self.circuit_id!r}, {self.num_qubits!r}, {self.label!r})"

    def inverse(self, annotated: bool = False):
        """Invert this :class:`HaiquCircuitGate`.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`HaiquCircuitdgGate`.

        Returns:
            HaiquCircuitdgGate: the inverted gate
        """
        return HaiquCircuitdgGate(
            circuit_id=self.circuit_id,
            num_qubits=self.num_qubits,
            label=self.label,
        )

    def validate_parameter(self, parameter):
        if isinstance(parameter, str):  # circuit_id
            return parameter
        elif isinstance(parameter, int):  # num_qubits
            return parameter
        elif isinstance(parameter, float):  # num_qubits
            return int(parameter)
        else:
            raise CircuitError(f"Invalid param type {type(parameter)} for gate {self.name}.")


class HaiquCircuitdgGate(HaiquCircuitGate):
    """
    The inverse of a :class:`HaiquCircuitGate`.

    .. note::

        Like :class:`HaiquCircuitGate`, this gate is a handle for a server-side circuit, and is expanded in the
        Haiqu cloud at submission time — see :class:`HaiquCircuitGate` for what works client-side and what
        requires :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.transpile` / :meth:`~haiqu.sdk.quantum_haiqu.Haiqu.run`.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────────────────────────────────────┐
        q_0: ┤                                       ├
        ...  ┤ Haiqucircuitdg(circuit_id,num_qubits) ├
        q_N  ┤                                       ├
             └───────────────────────────────────────┘
    """

    def __init__(
        self,
        circuit_id: str,
        num_qubits: int | float,
        *,
        label: str | None = None,
        name: str = "HaiquCircuitdg",
    ):
        """
        Args:
            circuit_id (str): The ID of the generated quantum circuit.
            num_qubits (int | float): The number of qubits on which the circuit operates.
            label (str | None): An optional label for identifying the instruction.
        """
        super().__init__(circuit_id, num_qubits, label=label, name=name)

    def inverse(self, annotated: bool = False):
        """Invert this :class:`HaiquCircuitdgGate`.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`HaiquCircuitGate`.

        Returns:
            HaiquCircuitGate: the inverted gate
        """
        return HaiquCircuitGate(
            circuit_id=self.circuit_id,
            num_qubits=self.num_qubits,
            label=self.label,
        )
