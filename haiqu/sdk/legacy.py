"""Deprecated functions that keep their earlier behaviour, available as ``haiqu.legacy``."""

import copy
import warnings
from typing import TYPE_CHECKING

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from . import errors
from .schemas import CircuitModel, DeviceModel, RunJobModel, RunJobType, RunSubmitModel
from .utils import (
    get_num_parameters_per_circuit,
    job_name_from_circuits,
    validate_and_normalize_parameters_and_observables,
    validate_error_mitigation_options,
    validate_readout_bases,
)

if TYPE_CHECKING:
    from .quantum_haiqu import Haiqu


def _warn_deprecated(name: str, replacement: str) -> None:
    """Warn that ``haiqu.legacy.<name>`` is temporary, attributing the warning to the user's call.

    The stack level skips this function, the legacy function and the friendly-error decorator
    wrapping it, so Python reports the line in the user's script or notebook cell.
    """
    warnings.warn(
        f"haiqu.legacy.{name} is deprecated and will be removed in a future release. It keeps the "
        f"behaviour {replacement} had before it returned a HybridJobModel, so that existing code keeps "
        f"working while it is migrated. Use {replacement} instead.",
        DeprecationWarning,
        stacklevel=4,
    )


class Legacy:
    """
    Deprecated versions of Haiqu functions whose behaviour has changed, kept temporarily so that
    existing code keeps working while it is migrated.

    Access them through the Haiqu object, for example ``haiqu.legacy``. They use the same login and
    current experiment as the rest of the Haiqu object, so no separate setup is needed.
    """

    def __init__(self, haiqu: "Haiqu") -> None:
        self._haiqu = haiqu

    def _check_login(self) -> None:
        """Log in the Haiqu object if needed; called by ``errors.graceful_api_errors_message``."""
        self._haiqu._check_login()

    @errors.graceful_api_errors_message
    def run(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit] | CircuitModel | list[CircuitModel],
        parameters: list | None = None,
        shots: int = 1000,
        observables: SparsePauliOp | list[SparsePauliOp] | list[list[SparsePauliOp]] | None = None,
        readout_bases: list[str] | None = None,
        device: DeviceModel | None = None,
        device_id: str | None = None,
        options: dict | None = None,
        use_mitigation: bool = False,
        use_packing: bool = False,
        pack_size: int | None = None,
        job_name: str | None = None,
        job_description: str | None = None,
        dry_run: bool = False,
    ) -> RunJobModel:
        """Run quantum circuits on the selected backend, as :meth:`Haiqu.run` did before it returned a hybrid job.

        .. deprecated:: 1.7.0

            ``haiqu.legacy.run`` is temporary, and every call emits a ``DeprecationWarning``. Use
            :meth:`Haiqu.run`, which returns a :class:`HybridJobModel`, instead. To keep existing code
            working while you migrate, replace ``haiqu.run(...)`` with ``haiqu.legacy.run(...)``: it takes
            the same arguments in the same order and returns the same :class:`RunJobModel`.

        **Migrating to** :meth:`Haiqu.run`:

        - ``haiqu.run`` takes a new ``auto_quality`` argument before ``job_name``. Pass ``job_name``,
          ``job_description`` and ``dry_run`` by keyword.
        - ``job.uncertainty`` becomes ``job.info["uncertainty"]``.
        - ``job.result_by_basis()`` becomes ``dict(zip(readout_bases, job.result()[i]))`` for circuit ``i``.
        - ``job.circuit_ids`` becomes ``job.input_circuit_ids``.
        - With ``use_mitigation=True``, ``haiqu.run`` applies different error mitigation defaults, so
          mitigated results may differ from those of ``haiqu.legacy.run``.

        This flexible method supports multiple execution scenarios, with different combinations of circuits, parameters, and
        observables. When multiple values are provided for any of them, the results are returned as nested lists with up to 3
        layers, ordered by circuits, then observables, and finally parameters.

        .. note::

            Every ``device_id`` — including ``"aer_simulator"`` — dispatches to the Haiqu cloud; there is no
            local execution path, and :meth:`Haiqu.login` must have succeeded first. Jobs submitted before
            :meth:`Haiqu.init` is called are logged under a shared default experiment; call
            ``haiqu.init("my experiment")`` first so runs stay organized on the dashboard.

        Args:
            circuits: The quantum circuit(s) to execute. Can be a single circuit or a list of circuits.
            parameters: The parameters for the circuits. Can be a single set of parameters or nested lists of parameter sets. For
                        multiple circuits, must be a list where each element corresponds to parameters for that circuit. Defaults
                        to ``None``, in which case the circuits must not have any parameters.

                        Each inner list must have length ``circuit.num_parameters``. Values are bound positionally in the
                        order of Qiskit's ``QuantumCircuit.parameters``. When parameters are created one-by-one, that order
                        is typically alphabetical by name (e.g. ``theta10`` before ``theta2``),
                        independent of gate addition order; ``ParameterVector`` circuits use vector
                        index order instead (e.g. ``theta[0]``, ``theta[1]``). Inspect
                        ``list(circuit.parameters)`` for the exact order.
            shots (int): The number of shots for each circuit execution. Defaults to 1000.
            observables: The observable(s) to measure. The order of Pauli terms in a single string follows the Qiskit
                         reversed-order convention (e.g., ``"IZ"`` measures qubit 0 in the Z basis). Defaults to ``None``,
                         in which case the circuits must include their own measurements.

                         Accepted shapes:

                         - **Single circuit:** a single ``SparsePauliOp``, the nested form
                           ``[[op1, op2, ...]]``, or a bare list ``[op1, op2, ...]``.
                         - **Multiple circuits:** a list of length ``num_circuits``, where each element is independently
                           either a single ``SparsePauliOp`` (one observable on that circuit) or a list of
                           ``SparsePauliOp`` (multiple observables on that circuit). Mixing is allowed —
                           ``[[op1, op2], op3]`` for two circuits is valid.

                         The fully-nested form is the unambiguous canonical shape and is recommended when the same code
                         path handles both single and multi-circuit submissions.
            readout_bases (list[str] | None): Per-qubit Pauli bases to read the circuits out in, e.g.
                ``["XXXX", "ZZZZ", "XZYZ"]``. Each string follows the same Qiskit reversed-order convention as
                ``observables`` (``basis[-1]`` applies to qubit 0) and needs one character per qubit, drawn from
                ``X``, ``Y`` and ``Z`` (use ``Z`` for a qubit that should not be rotated). The circuits keep their
                own measurements; this rotates the readout before them.

                One distribution is returned per basis per circuit, so ``result()`` gains a basis axis in the
                same position observables occupy: ``[circuit][basis]``, or ``[circuit][basis][parameter]`` with a
                parameter sweep. Use :meth:`RunJobModel.result_by_basis` to get them keyed by basis string.

                Mutually exclusive with ``observables``. Combines with ``use_packing``: each packed copy of
                the circuit is read out in the same basis, so a basis string still has one character per
                circuit qubit. Note that each basis is a separate execution, so the cost is
                ``len(readout_bases) * shots``. Defaults to ``None``, which reads out in the computational
                basis as before.
            device (DeviceModel | None): The device to run the circuits on. If specified, ``device_id`` is ignored.
                Devices from :meth:`Haiqu.get_device` with ``use_fractional_gates=True`` use fractional gates.
            device_id (str | None): The ID of the device to run the circuits on. Defaults to ``None``.
            options (dict | None): Options to pass to the device. Supports an optional
                ``"error_mitigation_options"`` key with a dictionary of boolean flags to control
                individual error mitigation components when ``use_mitigation=True``. Supported keys:

                - ``"dynamical_decoupling"`` (bool): Toggle dynamical decoupling. Defaults to ``True``.
                - ``"readout_mitigation"`` (bool): Toggle readout error mitigation. Defaults to ``True``.
                - ``"noise_tailoring"`` (bool): Toggle noise tailoring via Pauli twirling. Defaults to ``False``.
                - ``"advanced_mitigation"`` (bool): Toggle advanced mitigation. Defaults to ``True``.

                To use MPS simulation on ``aer_simulator``, pass ``{"method": "matrix_product_state"}``.

                An optional ``"noise_model"`` key runs a noisy simulation. It is only supported when
                ``device_id="aer_simulator"`` or ``device_id="ionq_simulator"``, and the accepted value
                differs by backend:

                - ``aer_simulator``: a ``dict`` or a ``qiskit_aer.noise.NoiseModel`` object (serialized
                  automatically before submission).
                - ``ionq_simulator``: a ``str`` identifier naming a characterized IonQ device
                  (e.g. ``"aria-1"``).

                Passing ``None`` (or omitting the key) runs a noiseless simulation.

                **Simulator qubit limits** (enforced server-side):

                - Statevector (``aer_simulator`` default): up to **24 qubits**.
                - MPS (``aer_simulator`` with ``method="matrix_product_state"``, including with fake
                  devices): no strict qubit limit.
                - Noisy simulation without MPS (fake devices such as ``fake_kyiv``, or
                  ``aer_simulator`` with a ``noise_model``): up to **12 qubits**.

            use_mitigation (bool): Whether to use error mitigation techniques. Defaults to ``False``.
            use_packing (bool): Whether to use circuit packing for efficient device utilization. Defaults to ``False``.
                **Warning:** Experimental — packing replicates circuits on unused device qubits
                to run multiple copies in parallel, which may increase errors for deeper input circuits.
                For example, a 4-qubit circuit with pack_size=2 and 1000 shots runs two copies
                in parallel with 500 shots each, yielding 1000 shots of results while only paying
                for 500 shot executions on the QPU — a 2x cost saving.
            pack_size (int | None): Number of circuit copies to pack onto the device. Must be >= 2.
                Only valid when ``use_packing=True``. If ``None`` (default), the backend will
                pack into at most 2/3 of the device qubits.
            job_name (str | None): The name for the job. If ``None`` (default), a name will be automatically generated.
            job_description (str | None): The description for the job.
            dry_run (bool): Whether to stop just prior to backend execution for QPU cost estimation. Defaults to ``False``.
                When ``True``, the job result will be empty since execution on the device is skipped.
                The estimated QPU cost is then available via ``job.estimated_qpu_cost``.

        Returns:
            RunJobModel: The Run job that will execute the circuit.
                Call ``job.result()`` to retrieve the execution results as a nested list ordered by
                *circuits → observables → parameters*:

                * Without observables: list of measurement distributions (``dict[str, float]``), one per
                  circuit, in Qiskit bit-order.
                * With observables, no parameter sweep: 2D list of expectation values, indexed ``[circuit][observable]``.
                * With observables and a parameter sweep: 3D list of expectation values, indexed
                  ``[circuit][observable][parameter]``.

                When ``dry_run=True``, ``result()`` is empty; use ``job.estimated_qpu_cost`` instead. ``job.info`` exposes
                auxiliary metadata (``uncertainty`` when observables are supplied, ``qpu_cost``).
                Run ``help(job.result)`` for the full description of result and ``info`` contents.

                In a notebook, a link to the job on the Dashboard is also displayed when
                ``haiqu.verbosity`` is ``"high"``.

        Raises:
            ValueError: If IBM Quantum rejects the IBM Quantum token and instance for a run on an IBM device, or
                cannot be reached to check them. Dry runs do not check them.

        Examples:
            Single circuit, no parameters, no observables:

            >>> from qiskit import QuantumCircuit
            >>> qc = QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> qc.measure_all()
            >>> job = haiqu.legacy.run(circuits=qc, device_id="aer_simulator")
            >>> job.result()  # Returns: [dist_c1] (bitstrings in Qiskit convention)
            [{'00': 0.504, '11': 0.496}]

            Single circuit, multiple parameters, multiple observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> theta = Parameter('θ')
            >>> qc = QuantumCircuit(2)
            >>> qc.ry(theta, 0)
            >>> qc.cx(0, 1)
            >>> params = [[0.5], [1.0]]
            >>> obs = [SparsePauliOp("ZZ"), SparsePauliOp("XX")]
            >>> job = haiqu.legacy.run(
            ...     circuits=qc,
            ...     parameters=params,
            ...     observables=obs,
            ...     device_id="aer_simulator",
            ... )
            >>> job.result()  # Returns: [[[exp_c1_obs1_p1, exp_c1_obs1_p2], [exp_c1_obs2_p1, exp_c1_obs2_p2]]]
            [[[1.0, 1.0], [0.49, 0.846]]]

            Multiple circuits, no parameters, no observables:

            >>> from qiskit import QuantumCircuit
            >>> qc1 = QuantumCircuit(2)
            >>> qc1.h(0)
            >>> qc1.cx(0, 1)
            >>> qc1.measure_all()
            >>> qc2 = QuantumCircuit(2)
            >>> qc2.x(0)
            >>> qc2.cx(0, 1)
            >>> qc2.measure_all()
            >>> job = haiqu.legacy.run(circuits=[qc1, qc2], device_id="aer_simulator")
            >>> job.result()  # Returns: [dist_c1, dist_c2]
            [{'11': 0.524, '00': 0.476}, {'11': 1.0}]
        """
        _warn_deprecated("run", "haiqu.run")
        haiqu = self._haiqu
        haiqu._check_experiment()
        circuits = haiqu._normalize_circuits(circuits)

        options = {} if options is None else copy.deepcopy(options)

        # Validate error_mitigation_options if provided
        validate_error_mitigation_options(options, use_mitigation)

        # Validate device ID
        if device is None:
            if device_id is None:
                raise ValueError("The `device` or `device_id` is required to run the circuits.")
            else:
                # Validate that device exists
                haiqu.get_device(device_id=device_id)

        # If both device and device_id are provided, device_id is ignored
        if device is not None:
            device_id = device.id

        haiqu._normalize_noise_model_option(device_id=device_id, options=options)

        # Validated before uploading anything, so a malformed basis never creates circuits.
        readout_bases = validate_readout_bases(readout_bases, observables, circuits)

        parameters, observables = validate_and_normalize_parameters_and_observables(
            parameters,
            observables,
            len(circuits),
            num_parameters_per_circuit=get_num_parameters_per_circuit(circuits),
        )

        # Validate and pass packing options
        haiqu._validate_packing(use_packing, pack_size)
        options["use_packing"] = use_packing
        if pack_size is not None:
            options["pack_size"] = pack_size

        if device is not None and device.use_fractional_gates:
            options["use_fractional_gates"] = True

        # The run endpoint takes device credentials inside `options`
        if not dry_run:
            haiqu._inject_device_credentials(device_id, options)

        logged_circuits = haiqu._log_circuits(circuits)

        if job_name is None:
            job_name = job_name_from_circuits(logged_circuits)

        # Submit the job
        job = haiqu._client.run(
            data=RunSubmitModel(
                experiment_id=haiqu._experiment.id,
                circuit_ids=[c.id for c in logged_circuits],
                parameters=parameters,
                shots=shots,
                observables=observables,
                readout_bases=readout_bases,
                device_id=device_id,
                options=options,
                use_mitigation=use_mitigation,
                name=job_name,
                description=job_description,
                dry_run=dry_run,
                run_type=RunJobType.DEVICE_RUN.value,
            )
        )
        if readout_bases is not None and job.readout_bases is None:
            # The submit model tolerates unknown fields, so an older deployment would silently
            # drop the bases and return computational-basis counts instead.
            raise RuntimeError(
                "This Haiqu API deployment does not support `readout_bases` (the server did not echo "
                "them back), so the returned counts would be in the computational basis. "
                "Please contact support."
            )
        haiqu._display_dashboard_links(job)
        return job

    @errors.graceful_api_errors_message
    def statevector_run(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit] | CircuitModel | list[CircuitModel],
        job_name: str | None = None,
        job_description: str | None = None,
    ) -> RunJobModel:
        """Run quantum circuits on a statevector simulator and obtain exact amplitudes of the wavefunctions, as
        :meth:`Haiqu.statevector_run` did before it returned a hybrid job.

        .. deprecated:: 1.7.0

            ``haiqu.legacy.statevector_run`` is temporary, and every call emits a ``DeprecationWarning``. Use
            :meth:`Haiqu.statevector_run`, which returns a :class:`HybridJobModel`, instead. To keep existing code
            working while you migrate, replace ``haiqu.statevector_run(...)`` with
            ``haiqu.legacy.statevector_run(...)``: it takes the same arguments and returns the same
            :class:`RunJobModel`.

        **Migrating to** :meth:`Haiqu.statevector_run`:

        - Final measurements are mandatory there (they are still ignored), so add ``qc.measure_all()`` to circuits
          without them.
        - ``job.circuit_ids`` becomes ``job.input_circuit_ids``.

        This execution type is restricted to non-parametrized circuits up to 20 qubits in size. Circuits may contain
        Haiqu gates, but no mid-circuit measurements or other logical operations. Final measurements in the circuit,
        if present, will be ignored. Statevector is measured over all qubits in their standard qiskit order.

        The returned amplitude ordering matches the indexing convention used by amplitude-based data loaders
        (:meth:`Haiqu.vector_loading`, :meth:`Haiqu.block_vector_loading`, :meth:`Haiqu.function_loading`,
        :meth:`Haiqu.distribution_loading`, :meth:`Haiqu.mps_loading`): index ``i`` corresponds to the
        computational-basis ket whose binary label has rightmost bit = qubit 0.

        Args:
            circuits: The quantum circuit(s) to execute. Can be a single circuit or a list of circuits.
            job_name (str | None): The name for the job. If ``None`` (default), a name will be automatically generated.
            job_description (str | None): The description for the job.

        Returns:
            RunJobModel: The Run job that will execute the circuit.
                Call ``job.result()`` to retrieve a list of complex-valued statevectors (one numpy array per input circuit),
                each of length ``2**num_qubits`` in standard Qiskit ordering (rightmost bit = qubit 0). Final measurements
                in the input circuits, if any, are ignored.
                Run ``help(job.result)`` for the full description of result and ``info`` contents.
                In a notebook, a link to the job on the Dashboard is also displayed when
                ``haiqu.verbosity`` is ``"high"``.

        Examples:
            Single circuit:

            >>> from qiskit import QuantumCircuit
            >>> qc = QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> qc.measure_all()  # measurements can be present or not
            >>> job = haiqu.legacy.statevector_run(qc)
            >>> job.result()  # Returns: [statevector]
            [array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j])]

            Multiple circuits:

            >>> from qiskit import QuantumCircuit
            >>> import numpy as np
            >>> qc_bell = QuantumCircuit(2)  # standard bell state
            >>> qc_bell.h(0)
            >>> qc_bell.cx(0, 1)
            >>> bell_gate_phased = haiqu.vector_loading([1, 0, 0, 1.j]).result()  # notice the imaginary amplitude
            >>> qc_bell_phased = QuantumCircuit(2)  # bell state with different phase on |11> state
            >>> qc_bell_phased.compose(bell_gate_phased, inplace=True)
            >>> job = haiqu.legacy.statevector_run([qc_bell, qc_bell_phased])
            >>> np.round(job.result(), 3)
            array([[ 0.707+0.j   ,  0.   +0.j   ,  0.   +0.j   ,  0.707+0.j   ],
                   [ 0.707+0.j   ,  0.   +0.j   ,  0.   +0.j   , -0.   +0.707j]])
        """
        _warn_deprecated("statevector_run", "haiqu.statevector_run")
        haiqu = self._haiqu
        haiqu._check_experiment()
        circuits = haiqu._normalize_circuits(circuits)
        logged_circuits = haiqu._log_circuits(circuits)

        if job_name is None:
            job_name = job_name_from_circuits(logged_circuits)

        # Submit the job
        job = haiqu._client.run(
            data=RunSubmitModel(
                experiment_id=haiqu._experiment.id,
                circuit_ids=[c.id for c in logged_circuits],
                name=job_name,
                description=job_description,
                run_type=RunJobType.STATEVECTOR_RUN.value,
            )
        )
        haiqu._display_dashboard_links(job)
        return job
