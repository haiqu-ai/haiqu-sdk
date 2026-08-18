"""
The Haiqu software development kit (SDK) is a collection of development tools to
accelerate the development of quantum applications with optimized application subroutines, cutting‑edge noise mitigation,
data loading, circuit compression and performance analytics tools.
"""

import json
import os
import copy
import configparser
import warnings
from typing import Any, Optional, Union
from collections.abc import Sequence
from inspect import signature
from numbers import Integral, Number, Real


from .optimization import (
    QUBO,
    SolverResult,
    cvar_expectation,
)
import numpy as np
import pandas as pd
import sympy

from qiskit import QuantumCircuit
from qiskit.circuit import Gate
from qiskit.circuit.library import UnitaryGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService

from requests.exceptions import HTTPError

from . import constants
from . import errors
from .api_client import ApiClient
from .backpropagation import backpropagation
from .hybrid import HybridProgram, layers
from .qml.compression_options import CompressionOptions
from .qml.problem import NonlinearVariationalProblem, VariationalProblem
from .utils import find_shared_parameters
from .qml.optimizer import NFTOptimizerOptions, OptimizerOptions
from .exceptions import (
    APIKeyRequiredError,
    CircuitNotRegisteredInExperimentError,
    ExperimentSearchByNameError,
    InvalidAPIKeyError,
    InvalidFiltersError,
    JobNotRegisteredInExperimentError,
)
from .errors import error_widget_or_string
from .version import get_version
from .schemas import (
    JOB_MODELS,
    ArtifactModel,
    PretrainingJobModel,
    PretrainingSubmitModel,
    StateCompressionJobModel,
    StateCompressionSubmitModel,
    Su2EquivariantCompilationJobModel,
    # StateCompressionEstimatesModel,
    # StateCompressionEstimatesSubmitModel,
    JobType,
    JobStatus,
    LocalJobModel,
    LocalJobSubmitModel,
    HybridSubmitModel,
    HybridJobModel,
    RunSubmitModel,
    RunJobModel,
    JobInsights,
    CircuitModel,
    CircuitSubmitModel,
    ExperimentSubmitModel,
    ExperimentUpdateModel,
    DataLoadingJobModel,
    DataLoadingSubmitModel,
    DataLoadingEstimatesModel,
    DeviceModel,
    SubmitMetricsModel,
    SubmitTranspilationModel,
    SubmitObservableBackpropagationModel,
    DataLoadingType,
    RunJobType,
    CompressionJobType,
    PretrainingJobType,
    UserModel,
    VariationalProblemSubmitModel,
    VariationalJobModel,
    PostprocessSKQDParams,
    SKQDSubmitModel,
    SKQDJobModel,
)
from .skqd import SKQDOptions
from .utils import (
    detect_notebook,
    detect_source_file,
    read_source_file,
    generate_artifact_name,
    get_circuit_hash,
    get_job_device,
    get_job_hash,
    get_job_name,
    get_job_results,
    get_job_shots,
    check_circuit_context,
    is_haiqu_generated,
    job_name_from_circuits,
    preprocess_metrics,
    to_qpy,
    from_qpy,
    validate_and_normalize_parameters_and_observables,
    get_num_parameters_per_circuit,
    AWS_DEFAULT_REGION,
    IQM_DEFAULT_SERVER_URL,
)
from .constants import (
    MAX_COMPRESSION_APPROXIMATION_LEVEL,
    MAX_COMPRESSION_QUBITS,
    MAX_COMPRESSION_TIME,
    MAX_DATA_LOADING_TIME,
    MAX_DL_EME_DENSITY,
    MAX_DL_FINE_TUNING_ITERATIONS,
    MAX_DL_LAYERS,
    MAX_DL_MPS_BOND_DIMENSION,
    MAX_DL_QUBITS,
    MAX_DL_VECTOR_LENGTH,
    MAX_DL_VECTOR_QUBITS,
    MIN_COMPRESSION_APPROXIMATION_LEVEL,
    MIN_DL_INTERVAL_WIDTH,
    SUPPORTED_MULTIVARIATE_DIMENSIONS,
)

__all__ = ["haiqu", "Haiqu"]


DEFAULT_EXPERIMENT = "Default Experiment"
LOGIN_ERROR_MESSAGE = """Valid Haiqu API key is required.
Please login with proper API key, use it as follows:

haiqu.login(api_access_key="...")

Or export OS environment variable `HAIQU_API_KEY` in the shell as follows:

export HAIQU_API_KEY="..."
"""


def format_docstring(**kwargs):
    def decorator(f):
        f.__doc__ = f.__doc__.format(**kwargs)

        return f

    return decorator


def check_parameters_match(f_other):
    def decorator(f):
        if signature(f).parameters != signature(f_other).parameters:
            raise TypeError(f"Different parameters in {f_other.__name__} and {f.__name__}")

        return f

    return decorator


def _prepare_nonlinear_problem(
    problem: Union[VariationalProblem, NonlinearVariationalProblem],
) -> tuple[QuantumCircuit, str, dict[str, tuple[list[str], list[float]]]]:
    """Normalize a pretraining/gradient problem to the nonlinear wire form.

    A linear ``VariationalProblem`` is wrapped into the trivial nonlinear objective ``"x"`` over a
    single observable, so both methods share one serialization path. Returns the ansatz, the loss
    expression string, and the observables map ``{symbol_name: ([term_strings], [coefficients])}``.
    """
    if isinstance(problem, VariationalProblem):
        problem = NonlinearVariationalProblem(problem.ansatz, "x", {"x": problem.observable})
    elif not isinstance(problem, NonlinearVariationalProblem):
        raise TypeError("problem must be a VariationalProblem or NonlinearVariationalProblem.")

    loss_expression = str(problem.loss)
    observables = {name: ([t for t, _ in pairs], [c for _, c in pairs]) for name, pairs in problem.observables.items()}
    return problem.ansatz, loss_expression, observables


class Haiqu:
    """
    High-level object to access all capabilities of the Haiqu quantum computing environment.

    AI agents: The agent orientation is installed to ``haiqu/sdk/AGENTS.md``, and can also be found at
    https://github.com/haiqu-ai/haiqu-sdk/blob/main/haiqu/sdk/AGENTS.md
    """

    def __init__(self) -> None:
        """Haiqu cloud runtime."""
        self._client = None
        self._experiment = None

        # shorthand to omit additional import in user code
        self.JobType = JobType

        self.version = get_version()

    def login(
        self,
        api_access_key: str | None = None,
        edge_uri: str | None = None,
        raise_on_error: bool = False,
    ) -> str:
        """Log in to the Haiqu cloud environment.

        Args:
            api_access_key (str | None): The token to access the Haiqu API. Optional in the Haiqu Lab environment. Defaults to
                                         ``None``, in which case the value will be taken from the ``HAIQU_API_KEY`` environment
                                         variable.
            edge_uri (str | None): The network location of the Haiqu API service. Defaults to ``None``, in which case the value
                                   is taken from the ``HAIQU_EDGE_URI`` environment variable if set, and otherwise from the
                                   default production endpoint.
            raise_on_error (bool): By default, human-friendly messages are returned. Set this to raise exceptions instead.

        Returns:
            str: Status message.

        Examples:
            >>> haiqu.login(api_access_key="HAIQU123")
            'Success: Welcome to the Quantum World, example@haiqu.ai!'
        """
        api_access_key = api_access_key or os.environ.get("HAIQU_API_KEY")
        edge_uri = edge_uri or os.environ.get("HAIQU_EDGE_URI", constants.REST_API_URI)
        if api_access_key is None:
            if raise_on_error:
                raise APIKeyRequiredError(LOGIN_ERROR_MESSAGE)
            return error_widget_or_string(LOGIN_ERROR_MESSAGE)
        self._client = ApiClient(
            api_access_key=api_access_key,
            rest_api_uri=edge_uri,
        )
        # Test the API Key and the connection to API service.
        res = self.user(raise_on_error=raise_on_error)
        if not isinstance(res, UserModel):
            # Drop client since auth failed
            self._client = None
            return res

        return f"Success: Welcome to the Quantum World, {res.username}!"

    def user(self, raise_on_error: bool = False) -> UserModel | str | None:
        """Get the logged in user.

        Args:
            raise_on_error (bool): By default, human-friendly messages are returned. Set this to raise exceptions instead.

        Returns:
            UserModel | str | None: The logged in user, or an error message or None if in Jupyter Lab.

        Examples:
            >>> haiqu.user()
            UserModel(email='example@haiqu.ai', ...)
        """
        if self._client is None:
            if raise_on_error:
                raise RuntimeError(LOGIN_ERROR_MESSAGE)
            return error_widget_or_string(LOGIN_ERROR_MESSAGE)

        try:
            user = self._client.get_user()
        except InvalidAPIKeyError as e:
            if raise_on_error:
                raise e
            return error_widget_or_string(LOGIN_ERROR_MESSAGE)
        except Exception as e:
            if raise_on_error:
                raise e
            return error_widget_or_string(str(e))
        return user

    @errors.graceful_api_errors_message
    def init(
        self,
        experiment_ctx: str,
        experiment_description: str = "",
        log_source_code: bool = False,
    ) -> str:
        """Set the current experiment (create or use existing own or shared experiment).

        If an experiment with this name doesn't exist yet, it will be created with the given description.

        Most objects (e.g. circuits) and actions (e.g. running a job) must be associated with an experiment. The default
        experiment will be used if one is not set explicitly.

        Args:
            experiment_ctx (str): The experiment name or ID of the shared experiment.
            experiment_description (str): An optional text description of the experiment. Only used when creating the experiment.
            log_source_code (bool): If True, the source code of the running script or notebook is read and sent to the
                Haiqu cloud as an experiment metric (``source_code``). Defaults to False. Enable this only if you are
                comfortable uploading your source to the server. When stored, the source code is available on the
                Dashboard so it can be accessed and shared among other users. Its visibility follows the experiment's:
                anyone who can see the experiment can see the stored source code, with no additional option to toggle.
                Every version is preserved, giving you and your collaborators the full history of the experiment's
                source code.

        Returns:
            str: Status message, URL to view the experiment on the dashboard.

        Examples:
            >>> haiqu.init("Example Experiment", "The experiment to use for all examples.")
            'Set current experiment to: Example Experiment. View on Dashboard: https://dashboard.haiqu.ai/experiments/...'

            Shared experiment can be set by its ID:

            >>> haiqu.init("exp-12345678-1234-5678-1234-567812345678")

            Opt in to uploading the running script/notebook source code to the Haiqu cloud:

            >>> haiqu.init("Example Experiment", log_source_code=True)
        """
        self._init(experiment_ctx, experiment_description, log_source_code=log_source_code)
        return (
            f"Set current experiment to: {self._experiment.name}. "
            f"View on Dashboard: {constants.DASHBOARD_EXPERIMENT_SCHEMA.format(experiment_id=self._experiment.id)}"
        )

    @errors.graceful_api_errors_message
    def update_experiment(self, name: str | None = None, description: str | None = None) -> str:
        """Update the current experiment metadata.
        Set the current experiment if wasn't set.

        Args:
            name (str | None): Updated experiment name. Defaults to the current name.
            description (str | None): Updated experiment description. Defaults to the current description.

        Returns:
            str: Status message.
        """
        if self._experiment is None:
            raise ValueError("No active experiment set. use haiqu.init() first.")

        if name is None and description is None:
            raise ValueError("One of the fields should be set: name or description.")

        self._experiment = self._client.update_experiment(
            experiment_id=self._experiment.id,
            data=ExperimentUpdateModel(
                name=name,
                description=description,
            ),
        )
        return f"Updated current experiment: {self._experiment.name}"

    @errors.graceful_api_errors_message
    def log(
        self,
        parent_ctx: CircuitModel | QuantumCircuit | Any = None,
        child_ctx: Any = None,
        name: str = None,
        description: str = None,
    ) -> str | CircuitModel:
        """Record data to the Haiqu cloud based on the context
        (e.g., circuit, or any other relevant information).

        This method functions similarly to a generic logger, in the same vein as
        the `log` method of the Weights & Biases Python SDK for machine learning.

        Args:
            parent_ctx: The input object to be logged.
            child_ctx: An optional child object, linked to the context of the parent.
            name: An optional name for the logged object.
            description: An optional description if logging a circuit.

        Returns:
            str | CircuitModel: Status message, circuit metadata.

        Examples:
            If used without parameters, ``haiqu.log()`` displays the nice widget
            with help. Try it out:

            >>> haiqu.log()

            This function always acts in the context of the current experiment.
            If the circuit metadata object (CircuitModel) is passed as the first
            argument, it will log data to that circuit.

            #### Log metrics

            >>> haiqu.log(12.34, name="Some value")
            >>> haiqu.log("Hello quantum world!", name="Some textual value")
            >>> haiqu.log([1, 2, 3], name="Experiment parameters")
            >>> # W&B style:
            >>> haiqu.log({"examples": ["one", "two", "three"]})
            >>> haiqu.log({"some_value": 12.34, "some_text": "Quantum!", "parameters": [1, 2, 3]})

            #### Log a circuit

            >>> from qiskit.circuit.random import random_circuit
            >>> qc = random_circuit(num_qubits=4, depth=1, max_operands=4, measure=True)
            >>> meta = haiqu.log(qc)
            >>> # or with name/description:
            >>> meta = haiqu.log(qc, name="Hello", description="World!")
            >>> # log artifact/metric to the circuit:
            >>> haiqu.log(meta, 42)

            #### Log the Matplotlib plt object

            >>> import matplotlib.pyplot as plt
            >>> plt.plot([1, 2, 3], [4, 5, 6], label="Label")
            >>> haiqu.log(plt, name="Awesome plot")
            >>> # or W&B style:
            >>> # haiqu.log({"chart": plt})

            #### Log the Pandas DataFrame

            >>> import pandas as pd
            >>> data = {
            >>>     "columns": [0, 1, 2],
            >>>     "data": [50, 40, 45]
            >>> }
            >>> df = pd.DataFrame(data)
            >>> haiqu.log(df, name="My DataFrame")

            #### Log the Drawer plot

            >>> from haiqu.sdk.wiz.drawer import Drawer
            >>> drawer = Drawer()
            >>> drawer.plot([1, 2, 3], [4, 5, 6])
            >>> haiqu.log(drawer, name="Cool drawer plot")

            #### Log the Matplotlib figure

            >>> import matplotlib.pyplot as plt

            >>> fig, ax = plt.subplots()
            >>> ax.plot([1, 2, 3], [4, 5, 6], label="Test Plot")
            >>> ...
            >>> haiqu.log(fig, name="Awesome figure")

            You can see logged data on Dashboard: https://dashboard.haiqu.ai
        """

        self._check_experiment()

        if isinstance(parent_ctx, dict):
            # Log objects the W&B way
            # haiqu.log({"some_value": 12.34, ...})
            self._log_experiment_metrics(**parent_ctx)
            return (
                f"Logged objects to the experiment {self._experiment.name!r}. "
                f"View on Dashboard: {constants.DASHBOARD_EXPERIMENT_SCHEMA.format(experiment_id=self._experiment.id)}"
            )

        if isinstance(parent_ctx, CircuitModel):
            # Circuit metadata object is the parent context
            # Log objects to this circuit
            circuit_meta = parent_ctx
            if child_ctx is not None:
                if isinstance(child_ctx, dict):
                    self._log_circuit_metrics(circuit=circuit_meta, **child_ctx)
                else:
                    name = name or generate_artifact_name(None, child_ctx)
                    self._log_circuit_metrics(circuit=circuit_meta, **{name: child_ctx})
                return f"Logged objects to the circuit {circuit_meta.name!r}."
            return circuit_meta

        circuit = check_circuit_context(parent_ctx)
        if circuit:
            # Circuit or QASM string is the parent context
            # Log the circuit and possibly metrics to it
            kwargs = {}
            if name is not None:
                kwargs["name"] = name
            if description is not None:
                kwargs["description"] = description
            circuit_meta = self._get_or_create_circuit(circuit=circuit, **kwargs)
            if child_ctx is not None:
                if isinstance(child_ctx, dict):
                    self._log_circuit_metrics(circuit=circuit, **child_ctx)
                else:
                    name = generate_artifact_name(None, child_ctx)
                    self._log_circuit_metrics(circuit=circuit, **{name: child_ctx})
            return circuit_meta

        if parent_ctx is None:
            raise errors.LogError("Please provide a valid object to log, e.g. a circuit or some data.")

        # Anything else is logged as experiment artifact/metric
        name = name or generate_artifact_name(parent_ctx, child_ctx)
        self._log_experiment_metrics(**{name: parent_ctx})
        return (
            f"Logged object to the experiment {self._experiment.name!r}. "
            f"View on Dashboard: {constants.DASHBOARD_EXPERIMENT_SCHEMA.format(experiment_id=self._experiment.id)}"
        )

    @errors.graceful_api_errors_message
    def compare_metrics(
        self,
        *circuits: QuantumCircuit | CircuitModel,
        help: bool = False,
        tiles_layout: bool = False,
    ) -> None:
        """Compare metrics of two circuits.

        Args:
            *circuits (QuantumCircuit | CircuitModel): The quantum circuits to compare.
            help (bool): If ``True``, include descriptions of the metrics in the output. Defaults to ``False``.
            tiles_layout (bool): If ``True``, use a tiled layout for better readability in Jupyter notebooks.
                Defaults to ``False``.

        Returns:
            None: Displays a comparison table of the metrics.

        Examples:
            >>> haiqu.compare_metrics(circuit1, circuit2, help=True)  # in Jupyter notebook
        """
        logged_circuits = []
        for c in circuits:
            if not isinstance(c, CircuitModel):
                c = self.log(c)
            c.wait_for_analytics()
            logged_circuits.append(c)

        return CircuitModel.compare_metrics(logged_circuits, help=help, tiles_layout=tiles_layout)

    @errors.graceful_api_errors_message
    def get_device(self, device_id: str, use_fractional_gates: bool = False) -> DeviceModel:
        """Get a device by name.

        Args:
            device_id (str): The name of the device.
            use_fractional_gates (bool): When this device is used for transpile/run, use IBM
                fractional gates (continuous-angle ``rx`` / ``rzz``, etc.). Only real IBM
                QPUs accept ``True``; fake/simulator/non-IBM raise. Defaults to ``False``.

        Returns:
            DeviceModel: The requested device. When ``use_fractional_gates=True``, the flag
                is stored client-side for :meth:`transpile`, :meth:`run`, and
                :meth:`variational_optimization`.

        Raises:
            ValueError: If the device is unknown, or ``use_fractional_gates=True`` is
                requested for a simulator, ``fake_*``, or non-IBM device.

        Examples:
            >>> haiqu.get_device("fake_torino")
            DeviceModel(...)

            >>> device = haiqu.get_device("ibm_boston", use_fractional_gates=True)
            >>> haiqu.transpile(circuit, device)
            >>> haiqu.run(circuit, device=device)
        """
        try:
            device = self._client.get_device(device_id=device_id)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ValueError(
                    f"The device `{device_id}` not found in the list of devices. "
                    "Please check input or use `haiqu.list_devices()` to get available devices"
                    "or `haiqu.list_simulators()` to get available simulators."
                ) from None
            raise e
        if use_fractional_gates and (
            device.simulator
            or device.vendor != "IBM"
            or not str(device.id).startswith("ibm_")
            or str(device.id).startswith("fake_")
        ):
            raise ValueError(
                "use_fractional_gates=True is only supported for real IBM Quantum devices. "
                f"Got device_id={device.id!r} (vendor={device.vendor!r}, simulator={device.simulator})."
            )
        # Copy so the client-only flag does not mutate a shared/catalog instance.
        return device.model_copy(update={"use_fractional_gates": use_fractional_gates})

    @errors.graceful_api_errors_message
    def transpile(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit] | CircuitModel | list[CircuitModel],
        device: DeviceModel,
        job_name: str | None = None,
        job_description: str | None = None,
        **transpilation_options: Any,
    ) -> CircuitModel | list[CircuitModel]:
        """Transpile a quantum circuit for a specific device.

        .. note::

            Transpilation runs in the Haiqu cloud. Circuits containing
            :class:`~haiqu.sdk.gates.HaiquCircuitGate` instances (for example, from data-loading job
            results or :meth:`~haiqu.sdk.schemas.CircuitModel.to_gate`) should be transpiled with this
            method because the gate definitions live server-side. The returned
            :class:`~haiqu.sdk.schemas.CircuitModel` carries cloud-computed metrics on
            ``.analytics`` (for example ``.analytics.depth``, ``.analytics.depth_2q``,
            ``.analytics.gates_2q``) or via ``.core_metrics()``.

        Args:
            circuits (QuantumCircuit | list[QuantumCircuit] | CircuitModel | list[CircuitModel]): The circuit(s) to transpile.
            device (DeviceModel): The target device for execution. If obtained with
                ``use_fractional_gates=True`` via :meth:`get_device`, fractional gates are used.
            job_name (str | None): The name for the job. If ``None`` (default), a name will be automatically generated.
            job_description (str | None): The description for the job.
            **transpilation_options: Additional arguments passed to the Qiskit transpiler. All parameters
                follow the Qiskit ``transpile()`` interface. A notable extension is ``seed_transpiler``:
                it accepts either a single integer (standard Qiskit behaviour) or a list of integers.
                When a list is provided, transpilation is run once per seed in parallel and the result
                with the lowest multi-qubit gate count is selected for each circuit.

        Returns:
            CircuitModel | list[CircuitModel]: The transpiled quantum circuit, logged to the current experiment.
                The returned ``CircuitModel`` will be linked to the original circuit.

        Examples:
            Haiqu SDK uses the Qiskit Transpiler for transpilation of your circuits.
            Transpiling a circuit is as easy as follows:

            >>> device = haiqu.get_device("fake_torino")
            >>> transpiled_circuit = haiqu.transpile(circuit, device)

            Every circuit stores the information about its transpiled versions, that can be viewed using:

            >>> haiqu.list_transpiled_circuits(circuit)

            It may sometimes be useful to compare the circuits resulting from transpilation with different parameters.

            >>> transpiled_circuit_opt0 = haiqu.transpile(circuit, device, optimization_level=0)
            >>> transpiled_circuit_opt3 = haiqu.transpile(circuit, device, optimization_level=3)
            >>> haiqu.compare_metrics(transpiled_circuit_opt0, transpiled_circuit_opt3)

            Pass a list of seeds to run multiple transpilations and automatically keep the best result
            (fewest two-qubit gates) for each circuit:

            >>> transpiled_circuit = haiqu.transpile(circuit, device, seed_transpiler=[0, 1, 2, 3, 4])

            Enable IBM fractional gates when targeting a real IBM QPU:

            >>> device = haiqu.get_device("ibm_boston", use_fractional_gates=True)
            >>> transpiled_circuit = haiqu.transpile(circuit, device)
        """
        self._check_experiment()
        logged_circuits = self._prepare_circuits(circuits)

        if not isinstance(device, DeviceModel):
            raise ValueError("The device must be a DeviceModel instance as returned by haiqu.get_device().")

        submit_data = SubmitTranspilationModel(
            experiment_id=self._experiment.id,
            circuit_ids=[c.id for c in logged_circuits],
            device_id=device.id,
            transpilation_options=transpilation_options,
            use_fractional_gates=bool(device.use_fractional_gates),
            name=job_name,
            description=job_description,
        )
        job = self._client.transpile_circuit(submit_data=submit_data)
        result = job.result()
        if len(result) == 1:
            return result[0]
        return result

    @errors.graceful_api_errors_message
    def list_experiments(
        self,
        widget: bool = True,
        pandas: bool = False,
    ) -> list | pd.DataFrame | None:
        """List available experiments.

        Args:
            widget (bool): If ``True`` (default), render the list as a Jupyter widget and return ``None``.
            pandas (bool): If ``True``, return the list as a Pandas DataFrame instead of a Python list. Defaults to ``False``.

        Returns:
            list | pandas.DataFrame | None: Experiments in a Python list or Pandas DataFrame, or ``None``.

        Examples:
            >>> haiqu.list_experiments()  # in Jupyter notebook
            >>> haiqu.list_experiments(widget=False)
            [Experiment 'Example Experiment', Experiment 'Another Experiment']
        """
        items = self._client.list_experiments()
        if pandas:
            return pd.DataFrame(c.model_dump() for c in items)
        elif widget:
            from haiqu.sdk.wiz.jupyter import list_experiments

            return list_experiments(items)
        return items

    @errors.graceful_api_errors_message
    def get_artifact(self, artifact_name: str) -> ArtifactModel:
        """Get an artifact by name how it was logged with haiqu.log().

        Args:
            artifact_name (str): The name of the artifact.

        Returns:
            ArtifactModel: The requested artifact.

        Examples:
            >>> haiqu.log(12.34, name="Some value")
            >>> haiqu.get_artifact("Some value")
            ArtifactModel(...)
        """
        self._check_experiment()
        try:
            return self._client.get_artifact(experiment_id=self._experiment.id, artifact_name=artifact_name)
        except HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                raise ValueError(
                    f"The artifact `{artifact_name}` not found for experiment `{self._experiment.name}`. "
                    "Please check input or use `haiqu.list_artifacts()` to get available artifacts. "
                    "Artifacts are logged with `haiqu.log()` and can be retrieved with `haiqu.get_artifact()`."
                ) from None
            raise e

    @errors.graceful_api_errors_message
    def list_artifacts(self, limit: int = 10, widget: bool = True, pandas: bool = False) -> list | pd.DataFrame | None:
        """List available artifacts in the current experiment.

        Args:
            limit (int): Limit the number of artifacts returned.
            widget (bool): If ``True`` (default), render the list as a Jupyter widget and return ``None``.
            pandas (bool): If ``True``, return the list as a Pandas DataFrame instead of a Python list. Defaults to ``False``.

        Returns:
            list | pandas.DataFrame | None: Artifacts in a Python list or Pandas DataFrame, or ``None``.

        Examples:
            >>> haiqu.list_artifacts()  # in Jupyter notebook
            >>> haiqu.list_artifacts(widget=False)
            [Artifact 'Example Artifact', Artifact 'Another Artifact']
        """
        self._check_experiment()
        items = self._client.list_artifacts(experiment_id=self._experiment.id, limit=limit)
        if pandas:
            return pd.DataFrame(c.model_dump() for c in items)
        elif widget:
            from haiqu.sdk.wiz.jupyter import list_artifacts

            return list_artifacts(items)
        return items

    @errors.graceful_api_errors_message
    def list_circuits(
        self,
        experiment_name: str | None = None,
        experiment_id: str | None = None,
        circuit_ids: list[str] | None = None,
        job_type: JobType | None = None,
        limit: int = 10,
        widget: bool = True,
        pandas: bool = False,
    ) -> list | pd.DataFrame | None:
        """List recent circuits.

        The circuits are filtered by experiment (name or ID, with the current experiment used if neither is specified), and
        limited to only the most recent few (10 by default).

        Args:
            experiment_name (str | None): Return circuits for the provided experiment name.
            experiment_id (str | None): Return circuits for the provided experiment ID.
            circuit_ids (list[str] | None): If not ``None``, return only circuits with these IDs.
            job_type (JobType | None): Return circuits for the provided job type in the experiment..
            limit (int): Limit the number of returned circuits.
            widget (bool): If ``True`` (default), render the list as a Jupyter widget and return ``None``.
            pandas (bool): If ``True``, return the list as a Pandas DataFrame instead of a Python list. Defaults to ``False``.

        Returns:
            list | pandas.DataFrame | None: Circuits in a Python list or Pandas DataFrame, or ``None``.

        Examples:
            Display a table with a few of the most recent circuits:

            >>> haiqu.list_circuits()  # in Jupyter notebook

            Get the most recent circuit in a specific experiment:

            >>> haiqu.list_circuits("Example Experiment", limit=1, widget=False)
            [Haiqu Circuit 'circuit-123']

            Get the most recent circuit for compression job type:

            >>> circuits = haiqu.list_circuits(job_type=haiqu.JobType.COMPRESSION, widget=False)

            Get the circuits for run job type:

            >>> circuits = haiqu.list_circuits(job_type=haiqu.JobType.RUN, widget=False)

            Filter circuits by their IDs:

            >>> filtered_circuits = haiqu.list_circuits(circuit_ids=["circ-123", "circ-456"], widget=False)
        """
        if experiment_name is None and experiment_id is None:
            self._check_experiment()
            experiment = self._experiment
        else:
            if experiment_name is not None and experiment_id is not None:
                raise InvalidFiltersError("Either experiment Name or ID must be specified, not both.")
            try:
                experiment = self._client.get_experiment(experiment_id=experiment_id, name=experiment_name)
            except HTTPError as e:
                if e.response is None or e.response.status_code == 404:
                    raise ExperimentSearchByNameError(
                        f"The experiment `{experiment_id or experiment_name}` not found in the list of experiments. "
                        "Please check input or use `haiqu.list_experiments()` to get available experiments. "
                        "You also can create new experiment with `haiqu.init()`."
                    ) from None
                raise e

        items = self._client.list_circuits(experiment_id=experiment.id, circuit_ids=circuit_ids, job_type=job_type, limit=limit)
        if pandas:
            return pd.DataFrame(c.model_dump() for c in items)
        elif widget:
            from haiqu.sdk.wiz.jupyter import list_circuits

            return list_circuits(items)
        return items

    @errors.graceful_api_errors_message
    def list_transpiled_circuits(
        self,
        circuit: CircuitModel,
        experiment_id: str | None = None,
        experiment_name: str | None = None,
        limit: int = 10,
        widget: bool = True,
        pandas: bool = False,
    ) -> list | None:
        """List recent transpiled circuits.
        The transpiled circuits are filtered by experiment ID (with the current experiment used if not specified), and limited to
        only the most recent few (10 by default). Transpiled circuits can also be optionally filtered by original circuit.

        Args:
            circuit (CircuitModel | str | None): If not ``None``, only show the transpiled circuits related
                to the given original circuit (as a ``CircuitModel`` or circuit ID).
            experiment_id (str | None): Return transpiled circuits for the provided experiment ID.
            experiment_name (str | None): Return transpiled circuits for the provided experiment name.
            limit (int): Limit the number of returned transpiled circuits.
            widget (bool): If ``True`` (default), render the list as a Jupyter widget and return ``None``.
            pandas (bool): If ``True``, return the list as a Pandas DataFrame instead of a Python list. Defaults to ``False``.

        Returns:
            list | None: Transpiled circuits in a Python list, or ``None``.

        Examples:
            Get the most recent transpiled circuit for a specific original circuit (possibly in a different experiment):
            >>> haiqu.list_transpiled_circuits(
            ...     "circ-abcdefab-cdef-abcd-efab-cdefabcdefab",
            ...     "exp-12345678-1234-5678-1234-567812345678",
            ...     limit=1,
            ...     widget=False,
            ... )
            [CircuitModel 'transp-circ-abcdefab-cdef-abcd-efab-cdefabcdefab-on-fake_torino']
        """
        if not isinstance(circuit, CircuitModel):
            raise ValueError("The circuit must be a CircuitModel instance as returned by haiqu.log() or haiqu.get_circuit().")

        if experiment_name is None and experiment_id is None:
            self._check_experiment()
            experiment = self._experiment
        else:
            if experiment_name is not None and experiment_id is not None:
                raise InvalidFiltersError("Either experiment Name or ID must be specified, not both.")
            try:
                experiment = self._client.get_experiment(experiment_id=experiment_id, name=experiment_name)
            except HTTPError as e:
                if e.response is None or e.response.status_code == 404:
                    raise ExperimentSearchByNameError(
                        f"The experiment `{experiment_id or experiment_name}` not found in the list of experiments. "
                        "Please check input or use `haiqu.list_experiments()` to get available experiments. "
                        "You also can create new experiment with `haiqu.init()`."
                    ) from None
                raise e

        circuit.update_from_backend()

        if circuit.transpiled_circuit_ids is None or len(circuit.transpiled_circuit_ids) == 0:
            return None

        items = []
        for transp_circuit_id in circuit.transpiled_circuit_ids[-limit:]:
            try:
                transp_circuit = self._client.get_circuit(experiment_id=experiment.id, circuit_id=transp_circuit_id)
                items.append(transp_circuit)
            except HTTPError:
                continue

        if pandas:
            return pd.DataFrame(c.model_dump() for c in items)
        elif widget:
            from haiqu.sdk.wiz.jupyter import list_transpiled_circuits

            return list_transpiled_circuits(items, title=f"Transpiled Circuits for {circuit.name}")
        return items

    @errors.graceful_api_errors_message
    def list_jobs(
        self,
        experiment_id: str | None = None,
        job_type: JobType | None = None,
        circuit: CircuitModel | str | None = None,
        limit: int = 10,
        widget: bool = True,
        pandas: bool = False,
    ) -> str | list | None:
        """List recent jobs, display them as a Jupyter widget (default) or return a Python list with job items, or
        a Pandas DataFrame with job data.

        Each job item contains the information about the job, input parameters, circuits, and results if the job is completed.
        For example, for a run job, the item will contain the information about the circuit that was run, the device it was
        run on, and the results of the execution if available. For transpilation jobs, the item will contain the original
        circuit(s), the target device, and the transpiled circuit(s). For compression jobs, the item will contain the
        original circuit, the compression type, and the compressed circuit.

        The jobs are filtered by experiment ID (with the current experiment used if not specified) and job type, and limited to
        only the most recent few (10 by default). Jobs can also be optionally filtered by circuit.

        Args:
            experiment_id (str | None): Return jobs for the provided experiment ID. If ``None``, return jobs for the
                                        current experiment.
            job_type (JobType | None): Filter jobs by type. Only jobs of the provided type will be returned.
                                       If ``None``, jobs of all types will be returned.
            circuit (CircuitModel | str | None): If not ``None``, only show the jobs related to the given circuit (as a
                                                 ``CircuitModel`` or circuit ID).
            limit (int): Limit the number of returned jobs.
            widget (bool): If ``True`` (default), render the list as a Jupyter widget and return ``None``.
            pandas (bool): If ``True``, return the list as a Pandas DataFrame instead of a Python list. Defaults to ``False``.

        Returns:
            list | pandas.DataFrame | None: Jobs in a Python list or Pandas DataFrame, or ``None`` if widget is rendered.

        Examples:
            Display a table with a few of the most recent jobs:

            >>> haiqu.list_jobs()  # will display a widget in Jupyter notebook

            Get Python list of the most recent jobs for the current experiment:

            >>> jobs = haiqu.list_jobs(widget=False)
            [RunJobModel 'Run job ABC',
             TranspilationJobModel 'Transpilation of 2 circuit(s) to device fake_torino',
             StateCompressionJobModel 'State Compression of Circuit-123',
             ...
            ]

            Get only state compression jobs for the current experiment:

            >>> jobs = haiqu.list_jobs(
            ...     job_type=haiqu.JobType.COMPRESSION,
            ...     widget=False
            ... )

            Get only transpilation jobs for the current experiment:

            >>> jobs = haiqu.list_jobs(
            ...     job_type=haiqu.JobType.TRANSPILATION,
            ...     widget=False
            ... )

            Get the most recent run job(s) for a specific circuit for a specific experiment:

            >>> haiqu.list_jobs(
            ...     experiment_id="exp-12345678-1234-5678-1234-567812345678",
            ...     job_type=haiqu.JobType.RUN,
            ...     circuit="circ-abcdefab-cdef-abcd-efab-cdefabcdefab",
            ...     widget=False,
            ... )
            [RunJobModel 'Run job 12345678-abcd-efab-1234-5678abcdefab'
             ...
            ]
        """
        if not experiment_id:
            self._check_experiment()
            experiment_id = self._experiment.id
        items = self._client.list_jobs(experiment_id=experiment_id, job_type=job_type, circuit=circuit, limit=limit)
        if pandas:
            return pd.DataFrame(c.model_dump() for c in items)
        elif widget:
            from haiqu.sdk.wiz.jupyter import list_jobs

            return list_jobs(items)
        return items

    @errors.graceful_api_errors_message
    def list_devices(
        self,
        widget: bool = True,
        pandas: bool = False,
    ) -> list | pd.DataFrame | None:
        """List devices for circuit transpilation and execution.

        Note that devices in the list may at times be offline or otherwise unavailable.

        Args:
            widget (bool): If ``True`` (default), render the list as a Jupyter widget and return ``None``.
            pandas (bool): If ``True``, return the list as a Pandas DataFrame instead of a Python list. Defaults to ``False``.

        Returns:
            list | pandas.DataFrame | None: List of devices, or ``None``.

        Examples:
            >>> haiqu.list_devices()  # in Jupyter notebook
            >>> haiqu.list_devices(widget=False)
            [DeviceModel(...), DeviceModel(...), ...]
        """
        items = []
        for device in self._client.list_devices():
            if not device.simulator:
                items.append(device)
        if pandas:
            return pd.DataFrame(item.model_dump() for item in items)
        if widget:
            from haiqu.sdk.wiz.jupyter import list_devices

            return list_devices(items)
        return items

    @errors.graceful_api_errors_message
    def list_simulators(
        self,
        widget: bool = True,
        pandas: bool = False,
    ) -> list | pd.DataFrame | None:
        """List available simulators for circuit execution.

        Args:
            widget (bool): If ``True`` (default), render the list as a Jupyter widget and return ``None``.
            pandas (bool): If ``True``, return the list as a Pandas DataFrame instead of a Python list. Defaults to ``False``.

        Returns:
            list | pandas.DataFrame | None: List of simulator names, or ``None``.

        Examples:
            >>> haiqu.list_simulators()  # in Jupyter notebook
            >>> haiqu.list_simulators(widget=False)
            [DeviceModel(...), DeviceModel(...), ...]
        """
        items = []
        for device in self._client.list_devices():
            if device.simulator:
                device.pending_jobs = None
                items.append(device)
        if pandas:
            return pd.DataFrame(item.model_dump() for item in items)
        if widget:
            from haiqu.sdk.wiz.jupyter import list_simulators

            return list_simulators(items)
        return items

    @errors.graceful_api_errors_message
    def get_circuit(self, circuit_id: str) -> CircuitModel:
        """Get a circuit by ID.

        Args:
            circuit_id (str): The ID of the circuit stored in the Haiqu cloud.

        Returns:
            CircuitModel: The requested circuit.

        Examples:
            >>> haiqu.get_circuit(circuit_id)
            Haiqu Circuit 'circuit-123'
        """
        try:
            circuit_metadata = self._client.get_circuit(circuit_id=circuit_id)
            return circuit_metadata
        except HTTPError as e:
            if e.response.status_code == 404:
                raise CircuitNotRegisteredInExperimentError(f"Circuit with given ID not found: {circuit_id}") from None
            raise e

    @errors.graceful_api_errors_message
    def get_job(self, job_id: str) -> JOB_MODELS:
        """Get a job by ID.

        Args:
            job_id (str): The ID of the job stored in the Haiqu cloud.

        Returns:
            JOB_MODELS: The requested job.

        Examples:
            >>> haiqu.get_job(job_id)
            LocalJobModel 'Local job 12345678-1234-5678-1234-567812345678'
            Run haiqu.list_jobs() to check its progress and access it by ID as
            haiqu.get_job("jb-abcdefab-cdef-abcd-efab-cdefabcdefab").
            Check further progress via: job.progress(), job.result()
        """
        try:
            job = self._client.get_job(job_id=job_id)
            return job
        except HTTPError as e:
            if e.response.status_code == 404:
                raise JobNotRegisteredInExperimentError(f"Job with given ID not found: {job_id}") from None
            raise e

    @staticmethod
    def _prepare_distribution_loading_params(
        num_qubits,
        distribution_name,
        interval_start,
        interval_end,
        loc,
        scale,
        encoding,
        num_layers,
        truncation_cutoff,
        name,
        shape,
    ):
        # Numeric limits and the accepted encodings are enforced by DataLoadingSubmitModel validation.
        if name is None:
            name = f"{distribution_name}({loc},{scale})[{interval_start},{interval_end}]"

        parameters = {
            "interval_start": interval_start,
            "interval_end": interval_end,
            "loc": loc,
            "scale": scale,
            "encoding": encoding,
            "num_layers": num_layers,
            "truncation_cutoff": truncation_cutoff,
        }
        parameters.update(shape)

        return name, parameters

    _AMPLITUDE_INDEXING_NOTE = """
        .. note::

            **Amplitude indexing (Qiskit little-endian).** The prepared state has ``2**num_qubits`` amplitudes.
            ``data[i]`` is the amplitude for computational-basis index ``i``, where the rightmost bit of ``i`` is
            qubit 0 (``q_0``). For example, with 2 qubits, ``data = [a, b, c, d]`` prepares
            ``a|00⟩ + b|01⟩ + c|10⟩ + d|11⟩``. Apply the returned gate with
            ``circuit.append(gate, range(num_qubits))`` so gate wire ``k`` acts on ``q_k``. Inspect amplitudes
            after loading with :meth:`statevector_run`.
    """

    _DISCRETIZATION_INDEXING_NOTE = """
        .. note::

            **Discretization order.** The interval ``[interval_start, interval_end]`` is divided into ``2**num_qubits``
            equal bins. Bin ``i`` (counting from 0) spans increasing position along the interval, and its midpoint
            value becomes amplitude index ``i`` using the same Qiskit little-endian basis indexing as
            :meth:`vector_loading` (rightmost bit of ``i`` is ``q_0``).
    """

    _BLOCK_VECTOR_INDEXING_NOTE = """
        .. note::

            **Block layout.** Flat ``data`` is split into consecutive blocks in list order. Each block is encoded with
            the same Qiskit little-endian amplitude indexing as :meth:`vector_loading`. Blocks are placed on
            consecutive qubit subsystems: block 0 uses the lowest-index qubits, block 1 the next qubit range, and so
            on. Apply the gate with ``circuit.append(gate, range(num_qubits))``.
    """

    _EME_FEATURE_ORDER_NOTE = """
        .. note::

            **Feature order (not amplitude indexing).** ``data[k]`` is the ``k``-th classical input feature in list
            order. This loader performs entangled manifold embedding; it does **not** map list entries to
            computational-basis amplitudes like :meth:`vector_loading`.
    """

    _MPS_QUBIT_ORDER_NOTE = """
        .. note::

            **Qubit / MPS site order.** Site tensor ``A_i`` (0-based) in the input MPS corresponds to circuit qubit
            ``q_i``. The resulting statevector uses standard Qiskit ordering (rightmost bit = qubit 0), the same
            convention as :meth:`vector_loading` and :meth:`statevector_run`.
    """

    _FOURIER_FREQUENCY_NOTE = """
        .. note::

            **Frequency layout.** ``fourier_coefficients[i0, i1, ...]`` is the weight of the harmonic
            ``exp(2j*pi*((min_freqs[0] + i0)*x0 + (min_freqs[1] + i1)*x1 + ...))``, so coefficients ascend in
            frequency along every axis — the layout :func:`~haiqu.sdk.utils.compute_fourier_coefficients` returns.

            ``np.fft.fftn`` does not produce this order: it wraps the negative frequencies onto the end, returning a
            length-4 axis as ``[0, 1, -2, -1]`` rather than ``[-2, -1, 0, 1]``. Pass a raw FFT result through
            ``np.fft.fftshift`` first, after which index 0 carries ``-(n // 2)`` — the ``min_freqs`` default.

            ``min_freqs`` is applied as a frequency-shift layer, not just as bookkeeping. A value that does not match
            how the coefficients were obtained encodes the wrong complex exponential, whose amplitude magnitudes may
            still look plausible. Slicing a narrower band out of a shifted spectrum moves the first frequency, so pass
            the ``min_freqs`` that :func:`~haiqu.sdk.utils.compute_fourier_coefficients` returns.
    """

    _FOURIER_INDEXING_NOTE = """
        .. note::

            **Grid indexing.** The encoded array is recovered from the statevector by reshaping it onto the per-dimension grid,
            ``Statevector(circuit).data.reshape([2**n for n in job.info["num_qubits_per_dimension"]])``, **up to a
            global phase** left behind by the frequency-shift layer. The first array dimension occupies the
            highest-index qubits, and grid point ``m`` along dimension ``d`` corresponds to
            ``x_d = m / 2**num_qubits_per_dimension[d]``.

            Unlike :meth:`vector_loading`, where qubits beyond the data size are dead zero padding, every output
            qubit above the per-dimension minimum performs band-limited Fourier interpolation onto a finer grid, at
            the cost of a Fourier layer rather than a wider state preparation. Resolution is therefore almost free:
            the same 9x9 block of coefficients reaches the same fidelity on a 32x32 grid (10 qubits) as on a 256x256
            grid (16 qubits). A dimension holding a single frequency needs no qubit to embed it,
            so its entry in ``num_qubits_per_dimension`` may be ``0`` when ``num_qubits`` is ``None``; such a
            dimension carries a constant, and giving it output qubits spreads that constant over its grid.
    """

    _MULTIVARIATE_DIMENSION_ORDER_NOTE = """
        .. note::

            **Dimension order (NumPy C ordering).** Dimensions occupy the statevector in C (row-major) order: the
            first dimension is the slowest-varying and takes the high-order qubits, the last dimension is the
            fastest-varying and takes the low-order qubits. With ``num_qubits=(n_0, n_1)`` the amplitudes reshape as
            ``np.reshape(statevector, (2**n_0, 2**n_1))``, so the grid point ``(i_0, i_1)`` is amplitude index
            ``i_0 * 2**n_1 + i_1``. Each dimension is discretized into ``2**n_k`` equal bins over its interval, with
            bin 0 at the interval start, and the bits within the statevector index follow the same Qiskit
            little-endian convention as :meth:`vector_loading`.
    """

    _distribution_loading_args = f"""
            num_qubits (int): The number of qubits in the generated circuit (from 1 to {MAX_DL_QUBITS} qubits).
            distribution_name (str): The name of the distribution. Can be any of the continuous distributions in
                `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>`__ or a
                `characteristic function <https://docs.haiqu.ai/catalog/characteristic_functions>`__.
            interval_start (Real): The beginning of the interval.
            interval_end (Real): The end of the interval. The interval width ``interval_end - interval_start`` must be at
                                 least {MIN_DL_INTERVAL_WIDTH}.
            loc (Real): The location to which to shift the distribution. Defaults to 0.
            scale (Real): The scaling factor by which to stretch the distribution (must be positive). Defaults to 1.
            encoding (str): How function values map to amplitudes. ``"probability"`` (default): amplitudes are proportional to
                            ``sqrt(f(x))``, so measurement probabilities are proportional to ``f(x)``. ``"amplitude"``:
                            amplitudes are proportional to ``f(x)``.
            num_layers (int): The number of layers in the generated circuit (from 1 to {MAX_DL_LAYERS} layers).
                              More layers can improve the quality of the output
                              distribution at the cost of a deeper circuit. Defaults to 1.
            truncation_cutoff (Real): The entanglement cutoff for later layers, from 0 to 1. Increasing this threshold may result
                                      in a smaller (but more approximate) circuit. Defaults to ``1e-6``.
            name (str | None): The name for the job and the produced circuit. If ``None`` (default), a name will be automatically
                               generated.
            job_description (str | None): The description for the job.
            **shape: Additional distribution parameters, required by some distributions. Refer to the distribution
                     documentation in `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`__ or the
                     `characteristic functions catalog <https://docs.haiqu.ai/catalog/characteristic_functions>`__
                     for more details.
    """

    @errors.graceful_api_errors_message
    @format_docstring(
        ARGS=_distribution_loading_args,
        AMP_INDEXING=_AMPLITUDE_INDEXING_NOTE,
        DISCRETIZATION=_DISCRETIZATION_INDEXING_NOTE,
    )
    def distribution_loading(
        self,
        num_qubits: int,
        distribution_name: str,
        interval_start: Real,
        interval_end: Real,
        loc: Real = 0,
        scale: Real = 1,
        encoding: str = "probability",
        num_layers: int = 1,
        truncation_cutoff: Real = 1e-6,
        name: str | None = None,
        job_description: str | None = None,
        **shape,
    ) -> DataLoadingJobModel:
        """Generate a quantum circuit that prepares a probability distribution.

        Given the description of a probability distribution function (PDF), this method creates a Data Loading job that runs in
        the Haiqu cloud. The result of this job is a circuit which can be used to supply the PDF to a quantum algorithm for
        processing. The cost and time of this job can be estimated with :meth:`distribution_loading_estimates`.

        The complexity of the generated circuit can be controlled by the ``num_layers`` and ``truncation_cutoff`` parameters.
        {AMP_INDEXING}
        {DISCRETIZATION}
        Args:{ARGS}
        Returns:
            DataLoadingJobModel: The Data Loading job that will generate the circuit for the probability distribution.
                Call ``job.result()`` to retrieve a Qiskit-compatible gate (``HaiquCircuitGate``) that prepares the requested
                probability distribution on ``num_qubits`` qubits. ``job.quality`` is the achieved state fidelity vs. the ideal
                target distribution; ``job.info`` exposes loader metadata (``fidelity``).
                Run ``help(job.result)`` for the full description of result and ``info`` contents.

        Examples:
            >>> num_qubits = 4
            >>> job = haiqu.distribution_loading(
            ...     num_qubits=num_qubits,
            ...     distribution_name="norm",
            ...     interval_start=-3,
            ...     interval_end=3,
            ...     name=f"Normal distribution ({{num_qubits}} qubits)",
            ... )
            >>> dl_gate = job.result()  # dl_gate is a Qiskit-compatible gate
            >>> fidelity = job.quality
            >>> print(f"Normal distribution was loaded with fidelity {{fidelity:.6f}}")
            Normal distribution was loaded with fidelity 0.999484
            >>> circuit = qiskit.QuantumCircuit(num_qubits)
            >>> circuit.append(dl_gate, range(num_qubits))
            >>> circuit.draw()
                 ┌────────────────────────────────────────────────────────────┐
            q_0: ┤0                                                           ├
                 │                                                            │
            q_1: ┤1                                                           ├
                 │  Haiqucircuit(circ-12345678-1234-5678-1234-567812345678,4) │
            q_2: ┤2                                                           ├
                 │                                                            │
            q_3: ┤3                                                           ├
                 └────────────────────────────────────────────────────────────┘
        """
        self._check_experiment()
        name, parameters = self._prepare_distribution_loading_params(
            num_qubits,
            distribution_name,
            interval_start,
            interval_end,
            loc,
            scale,
            encoding,
            num_layers,
            truncation_cutoff,
            name,
            shape,
        )

        # TODO: Add check for the required credits, use data_loading_estimates()
        # TODO: If user has less credits balance - inform the User and don't start a job.

        return self._client.data_loading(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.DISTRIBUTION_LOADING.value,
                name=name,
                description=job_description,
                experiment_id=self._experiment.id,
                num_qubits=num_qubits,
                distribution_name=distribution_name,
                parameters=parameters,
            )
        )

    @staticmethod
    def _prepare_multivariate_distribution_loading_params(
        num_qubits,
        distribution_name,
        interval,
        encoding,
        num_layers,
        truncation_cutoff,
        fine_tuning_iterations,
        distribution_params,
        copula_params,
        marginal_distribution_names,
        marginal_distribution_params,
        max_time,
        name,
    ):
        if isinstance(num_qubits, Integral):
            num_qubits = int(num_qubits)
            if num_qubits < SUPPORTED_MULTIVARIATE_DIMENSIONS:
                raise ValueError(
                    f"num_qubits must be at least {SUPPORTED_MULTIVARIATE_DIMENSIONS} for multivariate distribution loading"
                )
        else:
            num_qubits = list(num_qubits)
            if len(num_qubits) < SUPPORTED_MULTIVARIATE_DIMENSIONS or not all(
                isinstance(n, Integral) and n >= 1 for n in num_qubits
            ):
                raise ValueError("num_qubits must be an integer or a sequence of positive integers per dimension")
            num_qubits = [int(n) for n in num_qubits]

        if not isinstance(distribution_name, str) or not distribution_name:
            raise ValueError("distribution_name must be a non-empty string")

        interval = list(interval)
        is_shared_interval = all(isinstance(bound, Real) for bound in interval)
        intervals = [interval] if is_shared_interval else interval
        for bounds in intervals:
            if len(bounds) != 2:
                raise ValueError("interval must be a pair (start, end) or a sequence of per-dimension interval pairs")
            start, end = bounds
            if not isinstance(start, Real) or not isinstance(end, Real):
                raise ValueError(f"Invalid interval start/end ({start}, {end}).")
            if end - start < MIN_DL_INTERVAL_WIDTH:
                raise ValueError(f"Interval width must be at least {MIN_DL_INTERVAL_WIDTH}.")
        intervals = [[float(start), float(end)] for start, end in intervals]
        interval = intervals[0] if is_shared_interval else intervals

        if distribution_params is not None and not isinstance(distribution_params, dict):
            raise ValueError("distribution_params must be a dictionary")
        if copula_params is not None and not isinstance(copula_params, dict):
            raise ValueError("copula_params must be a dictionary")

        is_copula = distribution_name.endswith("_copula")
        if is_copula:
            if distribution_params:
                raise ValueError("distribution_params is only valid for direct multivariate distributions")
            if copula_params is None:
                raise ValueError("copula_params is required for copula loading")
            if marginal_distribution_names is None or marginal_distribution_params is None:
                raise ValueError("copula loading requires marginal_distribution_names and marginal_distribution_params")
            if isinstance(marginal_distribution_names, str):
                raise ValueError("marginal_distribution_names must be a sequence of distribution names")
            if len(marginal_distribution_names) != len(marginal_distribution_params):
                raise ValueError("copula loading requires matching marginal_distribution_names and marginal_distribution_params")
            if not all(isinstance(marginal, str) for marginal in marginal_distribution_names):
                raise ValueError("marginal_distribution_names must contain strings")
            if not all(isinstance(params, dict) for params in marginal_distribution_params):
                raise ValueError("marginal_distribution_params must contain dictionaries")
        else:
            if copula_params or marginal_distribution_names is not None or marginal_distribution_params is not None:
                raise ValueError(
                    "copula_params and marginal distributions are only valid when distribution_name is a copula, "
                    "whose name ends with '_copula'"
                )
            if not distribution_params:
                raise ValueError("distribution_params is required for a direct multivariate distribution")

        declared_dims = {}
        if not isinstance(num_qubits, int):
            declared_dims["num_qubits"] = len(num_qubits)
        if len(intervals) > 1:
            declared_dims["interval"] = len(intervals)
        if is_copula:
            declared_dims["marginal_distribution_names"] = len(marginal_distribution_names)
        if len(set(declared_dims.values())) > 1:
            raise ValueError(f"inconsistent number of dimensions across arguments: {declared_dims}")

        num_dimensions = next(iter(declared_dims.values()), SUPPORTED_MULTIVARIATE_DIMENSIONS)
        if num_dimensions != SUPPORTED_MULTIVARIATE_DIMENSIONS:
            raise ValueError(
                f"only {SUPPORTED_MULTIVARIATE_DIMENSIONS}-dimensional distributions are currently supported, "
                f"but the arguments describe {num_dimensions} dimensions: {declared_dims}"
            )
        if isinstance(num_qubits, int) and num_qubits % SUPPORTED_MULTIVARIATE_DIMENSIONS != 0:
            raise ValueError(
                f"a total num_qubits is split evenly across dimensions, so it must be a multiple of "
                f"{SUPPORTED_MULTIVARIATE_DIMENSIONS}; pass a per-dimension sequence for an uneven split"
            )

        parameters = {
            "num_qubits": num_qubits,
            "interval": interval,
            "encoding": encoding,
            "num_layers": num_layers,
            "truncation_cutoff": truncation_cutoff,
            "fine_tuning_iterations": fine_tuning_iterations,
            "max_time": max_time,
        }
        if distribution_params is not None:
            parameters["distribution_params"] = distribution_params
        if copula_params is not None:
            parameters["copula_params"] = copula_params
        if marginal_distribution_names is not None:
            parameters["marginal_distribution_names"] = list(marginal_distribution_names)
        if marginal_distribution_params is not None:
            parameters["marginal_distribution_params"] = list(marginal_distribution_params)

        if name is None:
            name = f"{distribution_name}({num_qubits} qubits)"

        total_num_qubits = num_qubits if isinstance(num_qubits, int) else sum(num_qubits)

        return name, parameters, total_num_qubits

    _multivariate_distribution_loading_args = f"""
            num_qubits (int | Sequence[int]): Either the total number of qubits, split evenly across the dimensions, or a
                                              sequence of per-dimension qubit counts such as ``(3, 1)`` for an uneven split.
                                              A total must therefore be a multiple of the number of dimensions.
            distribution_name (str): The name of a multivariate distribution (e.g. ``bivariate_gamma``, ``bivariate_student_t``,
                                     ``bivariate_von_mises``, ``marshall_olkin_weibull``, ``bivariate_poisson_normal_approx``)
                                     or a copula (``gaussian_copula``, ``clayton_copula``, ``gumbel_copula``,
                                     ``frank_copula``). Names ending in ``_copula`` select copula loading and require
                                     ``copula_params`` with marginal distributions instead of ``distribution_params``.
                                     See the
                                     `multivariate distributions catalog
                                     <https://docs.haiqu.ai/catalog/multivariate_distributions>`__ for the full list.
            interval (Sequence): The spatial domain. Either a single pair ``(a, b)`` shared by all dimensions, or a sequence
                                 of per-dimension interval pairs. Each interval width ``b - a`` must be at least
                                 {MIN_DL_INTERVAL_WIDTH}. Defaults to ``(0, 1)``.
            encoding (str): How the pdf maps to amplitudes. ``"probability"`` (default): amplitudes are proportional to
                            ``sqrt(pdf)``, so measurement probabilities are proportional to the pdf. ``"amplitude"``:
                            amplitudes are proportional to the pdf.
            num_layers (int): The number of layers in the generated circuit (from 1 to {MAX_DL_LAYERS} layers). More layers can
                              improve the quality of the output distribution at the cost of a deeper circuit. Defaults to 1.
            truncation_cutoff (Real): The entanglement cutoff for later layers, from 0 to 1. Increasing this threshold may result
                                      in a smaller (but more approximate) circuit. Defaults to ``1e-6``.
            fine_tuning_iterations (int): The number of the gradient-based fine-tuning iterations applied after analytical
                                          compilation. Defaults to 20, maximal is {MAX_DL_FINE_TUNING_ITERATIONS}.
            distribution_params (dict | None): Keyword arguments of a direct multivariate distribution (e.g.
                                               ``{{"nu": 2.0, "rho": 0.5}}`` for ``bivariate_gamma``). Only valid for direct
                                               multivariate distributions.
            copula_params (dict | None): Keyword arguments of the named copula (e.g. ``{{"rho": 0.5}}`` for
                                         ``gaussian_copula``). Only valid for copulas.
            marginal_distribution_names (Sequence[str] | None):
                `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>`__ 1D
                distribution names used as copula marginals, e.g. ``["norm", "gamma"]``. Only valid for copulas.
            marginal_distribution_params (Sequence[dict] | None): Dictionaries of ``scipy.stats`` marginal parameters, in the
                                                                  same order as ``marginal_distribution_names``.
            max_time (int | float): Soft time limit for the job (in seconds). The job will first always produce the initial
                                    result and then limit the fine-tuning stage by the remaining time left.
                                    Defaults to {MAX_DATA_LOADING_TIME} ({MAX_DATA_LOADING_TIME//60} min).
            name (str | None): The name for the job and the produced circuit. If ``None`` (default), a name will be
                               automatically generated.
            job_description (str | None): The description for the job.
    """

    @errors.graceful_api_errors_message
    @format_docstring(ARGS=_multivariate_distribution_loading_args, DIMENSION_ORDER=_MULTIVARIATE_DIMENSION_ORDER_NOTE)
    def multivariate_distribution_loading(
        self,
        num_qubits: int | Sequence[int],
        distribution_name: str,
        interval: Sequence = (0, 1),
        encoding: str = "probability",
        num_layers: int = 1,
        truncation_cutoff: Real = 1e-6,
        fine_tuning_iterations: int = 20,
        distribution_params: dict | None = None,
        copula_params: dict | None = None,
        marginal_distribution_names: Sequence[str] | None = None,
        marginal_distribution_params: Sequence[dict] | None = None,
        max_time: int | float = MAX_DATA_LOADING_TIME,
        name: str | None = None,
        job_description: str | None = None,
    ) -> DataLoadingJobModel:
        """Generate a quantum circuit that prepares a multivariate probability distribution or a copula-based joint PDF.

        Given the description of a multivariate probability distribution function (PDF), this method creates a Data Loading job
        that runs in the Haiqu cloud. The result of this job is a circuit which can be used to supply the joint PDF to a quantum
        algorithm for processing. For one-dimensional
        `scipy.stats <https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions>`__ distributions use
        :meth:`distribution_loading`. See the
        `multivariate distributions catalog <https://docs.haiqu.ai/catalog/multivariate_distributions>`__ for the full list
        of supported distributions and copulas.

        Note:
            Joint distributions of two variables are supported at present, so ``num_qubits``, ``interval`` and the copula
            marginals must describe two dimensions.
        {DIMENSION_ORDER}
        Args:{ARGS}
        Returns:
            DataLoadingJobModel: The Data Loading job that will generate the circuit for the joint PDF.
                Call ``job.result()`` to retrieve a Qiskit-compatible gate (``HaiquCircuitGate``) that prepares the requested
                distribution. ``job.quality`` is the circuit-to-MPS compilation fidelity.

        Examples:
            >>> job = haiqu.multivariate_distribution_loading(
            ...     num_qubits=8,
            ...     distribution_name="bivariate_gamma",
            ...     interval=((0, 15), (0, 15)),
            ...     distribution_params={{"nu": 2.0, "rho": 0.5}},
            ... )
            >>> dl_gate = job.result()  # dl_gate is a Qiskit-compatible gate

            Copula with explicit marginals:

            >>> job = haiqu.multivariate_distribution_loading(
            ...     num_qubits=8,
            ...     distribution_name="gaussian_copula",
            ...     interval=((-3, 3), (0, 10)),
            ...     copula_params={{"rho": 0.5}},
            ...     marginal_distribution_names=["norm", "gamma"],
            ...     marginal_distribution_params=[{{"loc": 0, "scale": 1}}, {{"a": 2.0}}],
            ... )
        """
        self._check_experiment()
        name, parameters, total_num_qubits = self._prepare_multivariate_distribution_loading_params(
            num_qubits,
            distribution_name,
            interval,
            encoding,
            num_layers,
            truncation_cutoff,
            fine_tuning_iterations,
            distribution_params,
            copula_params,
            marginal_distribution_names,
            marginal_distribution_params,
            max_time,
            name,
        )

        return self._client.data_loading(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.MULTIVARIATE_DISTRIBUTION_LOADING.value,
                name=name,
                description=job_description,
                experiment_id=self._experiment.id,
                num_qubits=total_num_qubits,
                distribution_name=distribution_name,
                parameters=parameters,
            )
        )

    @staticmethod
    def _prepare_vector_loading_params(
        data,
        num_layers,
        truncation_cutoff,
        fine_tuning_iterations,
        max_time,
        name,
    ):
        # Scalar limits are enforced by DataLoadingSubmitModel schema validation. The vector length
        # limit depends on the materialized array, so it is checked here on the client side.
        parameters = {
            "data": np.array(data),
            "num_layers": num_layers,
            "truncation_cutoff": truncation_cutoff,
            "fine_tuning_iterations": fine_tuning_iterations,
            "max_time": max_time,
        }

        if not 1 <= len(parameters["data"]) <= MAX_DL_VECTOR_LENGTH:
            raise ValueError(f"Vector length must be between 1 and {MAX_DL_VECTOR_LENGTH}.")

        if name is None:
            name = f"VectorLoading(size:{len(parameters['data'])})"

        return name, parameters

    _vector_loading_args = f"""
            data (Sequence[Number]): The vector with data to encode (length of data is from 1 to ``2**{MAX_DL_VECTOR_QUBITS}``
                                     values). ``data[i]`` is the amplitude for basis index ``i`` in Qiskit little-endian order
                                     (rightmost bit of ``i`` is qubit 0).
            num_qubits: (int | None): The number of qubits in the generated circuit (from 1 to {MAX_DL_VECTOR_QUBITS} qubits).
                                      If ``None`` (default), it is set automatically from the size of the data.
            num_layers (int): The number of layers in the generated circuit (from 1 to {MAX_DL_LAYERS} layers).
                              More layers can improve the quality of the output
                              vector at the cost of a deeper circuit. Defaults to 2.
            truncation_cutoff (Real): The entanglement cutoff for later layers, from 0 to 1. Increasing this threshold may result
                                      in a smaller (but more approximate) circuit. Defaults to ``1e-6``.
            fine_tuning_iterations (int): The maximum number of fine-tuning iterations to perform after each layer is added.
                                          Increasing this limit may improve the quality of the circuit by using more classical
                                          resources. Defaults to 20, maximal is {MAX_DL_FINE_TUNING_ITERATIONS}.
            max_time (int | float): Soft time limit for the job (in seconds).
                            The data loading job will first always produce the initial result and then limit the fine-tuning
                            stage by the remaining time left. If time limit exceeds during the fine-tuning - the best
                            current result will be returned. Defaults to {MAX_DATA_LOADING_TIME}
                            ({MAX_DATA_LOADING_TIME // 60} min). Max allowed job time is {MAX_DATA_LOADING_TIME // 60} min.
                            The job can take more wall clock time than user specified `max_time` due to latency,
                            initialization overheads or if the initial result already takes more time.
            name (str | None): The name for the job and the produced circuit. If ``None`` (default), a name will be automatically
                               generated.
    """

    @errors.graceful_api_errors_message
    @format_docstring(ARGS=_vector_loading_args, AMP_INDEXING=_AMPLITUDE_INDEXING_NOTE)
    def vector_loading(
        self,
        data: Sequence[Number],
        num_qubits: int | None = None,
        num_layers: int = 2,
        truncation_cutoff: Real = 1e-6,
        fine_tuning_iterations: int = 20,
        max_time: int | float = MAX_DATA_LOADING_TIME,
        name: str | None = None,
        job_description: str | None = None,
    ) -> DataLoadingJobModel:
        """Generate a quantum circuit that prepares an arbitrary real or complex vector.

        Given a vector of data, this method creates a Data Loading job that runs in the Haiqu cloud. The result of this job is a
        circuit which can be used to supply the vector to a quantum algorithm for processing. The cost and time of this job can be
        estimated with :meth:`vector_loading_estimates`.

        The complexity and quality of the generated circuit can be controlled by the ``num_layers``, ``truncation_cutoff``, and
        ``fine_tuning_iterations`` parameters.

        If ``len(data) < 2**num_qubits``, the vector will be padded with zeros.
        {AMP_INDEXING}
        .. note::

            Multiple calls with identical inputs will generate multiple circuits. To save time, persist the job ID together with
            your results rather than re-running the loading job.

        Args:{ARGS}
        Returns:
            DataLoadingJobModel: The Data Loading job that will generate the circuit for the data vector.
                Call ``job.result()`` to retrieve a Qiskit-compatible gate (``HaiquCircuitGate``) that prepares the input data
                vector. ``job.quality`` is the achieved state fidelity vs. the ideal target vector; ``job.info`` exposes loader
                metadata (``fidelity``).
                Run ``help(job.result)`` for the full description of result and ``info`` contents.

        Examples:
            >>> bell_state = [1, 0, 0, 1]  # normalization is not required
            >>> job = haiqu.vector_loading(data=bell_state, name="Bell state Vector Loading")
            >>> vl_gate = job.result()  # vl_gate is a Qiskit-compatible gate
            >>> fidelity = job.quality
            >>> print(f"Bell state was loaded with fidelity {{fidelity:.6f}}")
            Bell state was loaded with fidelity 1.000000
            >>> print(f"Vector loading required {{job.num_qubits}} qubits")
            Vector loading required 2 qubits
            >>> circuit = qiskit.QuantumCircuit(job.num_qubits)
            >>> circuit.append(vl_gate, range(job.num_qubits))
            >>> circuit.draw()
                 ┌────────────────────────────────────────────────────────────┐
            q_0: ┤0                                                           ├
                 │  Haiqucircuit(circ-12345678-1234-5678-1234-567812345678,2) │
            q_1: ┤1                                                           ├
                 └────────────────────────────────────────────────────────────┘
        """
        self._check_experiment()
        name, parameters = self._prepare_vector_loading_params(
            data, num_layers, truncation_cutoff, fine_tuning_iterations, max_time, name
        )

        return self._client.data_loading(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.VECTOR_LOADING.value,
                name=name,
                description=job_description,
                num_qubits=num_qubits,
                experiment_id=self._experiment.id,
                parameters=parameters,
            )
        )

    @staticmethod
    def _prepare_block_vector_loading_params(
        data,
        num_blocks,
        target_num_qubits,
        overlap,
        num_layers,
        truncation_cutoff,
        fine_tuning_iterations,
        max_time,
        name,
    ):
        # Numeric limits are enforced by DataLoadingSubmitModel schema validation.
        parameters = {
            "data": np.array(data),
            "num_blocks": num_blocks,
            "target_num_qubits": target_num_qubits,
            "overlap": overlap,
            "num_layers": num_layers,
            "truncation_cutoff": truncation_cutoff,
            "fine_tuning_iterations": fine_tuning_iterations,
            "max_time": max_time,
        }

        if name is None:
            size = len(parameters["data"]) if isinstance(data, Sequence) else "unknown"
            name = f"BlockVectorLoading(size:{size})"

        return name, parameters

    _block_vector_loading_args = f"""
            data (Sequence[Number] | Sequence[Sequence[Number]]): The vector or matrix with data to encode.
            num_blocks (int | Sequence[int] | None): The number of blocks into which to split the data. It must be a single number
                                                     in one dimension and a pair of numbers (rows and columns) in two dimensions.
                                                     If ``None`` (default), the number of blocks is inferred from
                                                     ``target_num_qubits``, which must be specified.
                                                     In result each block must be of size not larger than {MAX_DL_VECTOR_QUBITS}
                                                     qubits.
            target_num_qubits (int | None): The qubit budget to assume when automatically determining the number of blocks. If
                                            ``None`` (default), the number of qubits depends on ``num_blocks``, which must be
                                            specified.
            overlap (int | float | None): The overlap blocks have with each other.
                                          An integer indicates the exact number of overlapping indices between consecutive blocks.
                                          A float in [0, 1) indicates fractional overlap between consecutive blocks.
                                          If ``None`` (default), the blocks do not overlap.
            num_layers (int): The number of layers in the generated circuit (from 1 to {MAX_DL_LAYERS} layers).
                              More layers can improve the quality of the circuit
                              blocks at the cost of a deeper circuit. Defaults to 2.
            truncation_cutoff (Real): The entanglement cutoff for later layers, from 0 to 1. Increasing this threshold may result
                                      in a smaller (but more approximate) circuit. Defaults to ``1e-6``.
            fine_tuning_iterations (int): The maximum number of fine-tuning iterations to perform after each layer is added.
                                          Increasing this limit may improve the quality of the circuit by using more classical
                                          resources. Defaults to 20, maximal is {MAX_DL_FINE_TUNING_ITERATIONS}.
            max_time (int | float): Soft time limit for the job (in seconds).
                            The data loading job will first always produce the initial result and then limit the fine-tuning
                            stage by the remaining time left. If time limit exceeds during the fine-tuning - the best
                            current result will be returned. Defaults to {MAX_DATA_LOADING_TIME}
                            ({MAX_DATA_LOADING_TIME // 60} min). Max allowed job time is {MAX_DATA_LOADING_TIME // 60} min.
                            The job can take more wall clock time than user specified `max_time` due to latency,
                            initialization overheads or if the initial result already takes more time. This time limit
                            will be evenly split across generation of each block.
            name (str | None): The name for the job and the produced circuit. If ``None`` (default), a name will be automatically
                               generated.
    """

    @errors.graceful_api_errors_message
    @format_docstring(ARGS=_block_vector_loading_args, BLOCK_INDEXING=_BLOCK_VECTOR_INDEXING_NOTE)
    def block_vector_loading(
        self,
        data: Sequence[Number] | Sequence[Sequence[Number]],
        num_blocks: int | Sequence[int] | None = None,
        target_num_qubits: int | None = None,
        overlap: float | int | None = None,
        num_layers: int = 2,
        truncation_cutoff: Real = 1e-6,
        fine_tuning_iterations: int = 20,
        max_time: int | float = MAX_DATA_LOADING_TIME,
        name: str | None = None,
        job_description: str | None = None,
    ) -> DataLoadingJobModel:
        """Generate a block-wise quantum circuit that prepares an arbitrary vector or matrix.

        Given a vector or matrix of real or complex data, this method creates a Data Loading job that runs in the Haiqu cloud. The
        result of this job is a circuit which can be used to supply the data to a quantum algorithm for processing.

        Unlike :meth:`vector_loading`, which uses the fewest qubits possible to encode the data, the block-wise strategy in
        :meth:`block_vector_loading` trades circuit depth for width. If additional qubits are available, they can be exploited to
        split the problem into several blocks, each of which is simpler. This reduces the overall depth of the circuit, making it
        more amenable to execution on noisy devices.

        Exactly one of ``num_blocks`` and ``target_num_qubits`` must be specified, which will determine how the vector or matrix
        is decomposed into blocks.

        The complexity and quality of the generated circuit can be controlled by the ``num_layers``, ``truncation_cutoff``, and
        ``fine_tuning_iterations`` parameters.
        {BLOCK_INDEXING}
        Args:{ARGS}
        Returns:
            DataLoadingJobModel: The Data Loading job that will generate the block-wise circuit for the data.
                Call ``job.result()`` to retrieve a Qiskit-compatible gate (``HaiquCircuitGate``) implementing the block-wise
                data preparation. ``job.quality`` is the achieved (mean) state fidelity across blocks; ``job.info`` exposes
                per-block metadata (``num_blocks``, ``num_qubits_per_block``, ``fidelity_per_block``, ``mean_fidelity``,
                ``global_fidelity``).
                Run ``help(job.result)`` for the full description of result and ``info`` contents.

        Examples:
            >>> vector = [0.5, 0.2, 1, 14, 0.3, 5, 0.2, 0.6]  # 8 elements (will split into 2 two-qubit blocks)
            >>> job = haiqu.block_vector_loading(data=vector, num_blocks=2, name="Block Vector Loading")
            >>> bvl_gate = job.result()  # bvl_gate is a Qiskit-compatible gate
            >>> fidelity = job.quality
            >>> print(f"Block vector was loaded with average fidelity {{fidelity:.6f}}")
            Block vector was loaded with average fidelity 1.000000
            >>> print(f"Block vector loading used {{job.num_qubits}} qubits")
            Block vector loading used 4 qubits
            >>> circuit = qiskit.QuantumCircuit(job.num_qubits)
            >>> circuit.append(bvl_gate, range(job.num_qubits))
            >>> circuit.draw()
                 ┌────────────────────────────────────────────────────────────┐
            q_0: ┤0                                                           ├
                 │                                                            │
            q_1: ┤1                                                           ├
                 │  Haiqucircuit(circ-12345678-1234-5678-1234-567812345678,4) │
            q_2: ┤2                                                           ├
                 │                                                            │
            q_3: ┤3                                                           ├
                 └────────────────────────────────────────────────────────────┘
        """
        self._check_experiment()
        name, parameters = self._prepare_block_vector_loading_params(
            data, num_blocks, target_num_qubits, overlap, num_layers, truncation_cutoff, fine_tuning_iterations, max_time, name
        )
        return self._client.data_loading(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.BLOCK_VECTOR_LOADING.value,
                name=name,
                description=job_description,
                experiment_id=self._experiment.id,
                parameters=parameters,
            )
        )

    @staticmethod
    def _prepare_entangled_manifold_embedding_params(
        data,
        density,
        real,
        periodicity,
        num_layers,
        truncation_cutoff,
        fine_tuning_iterations,
        max_time,
        name,
    ):
        # Numeric limits are enforced by DataLoadingSubmitModel schema validation.
        parameters = {
            "data": np.array(data),
            "density": density,
            "real": real,
            "periodicity": periodicity,
            "num_layers": num_layers,
            "truncation_cutoff": truncation_cutoff,
            "fine_tuning_iterations": fine_tuning_iterations,
            "max_time": max_time,
        }

        if name is None:
            name = f"EntangledManifoldEmbedding(size:{len(parameters['data'])},density:{density})"

        return name, parameters

    _entangled_manifold_embedding_args = f"""
                data (Sequence[Real]): The real vector with data to encode. ``data[k]`` is the ``k``-th input feature in list
                                       order (not a computational-basis amplitude index).
                density: (int | None): Feature density of the encoding (from 1 to {MAX_DL_EME_DENSITY}). Larger values result in
                                       more features encoded per qubit but resulting quantum states are more entangled. Ignored if
                                       ``num_qubits`` is set, in which case the minimal density that is compatible
                                       with the given number of qubits is chosen. Defaults to ``2``.
                num_qubits: (int | None): number of qubits for the embedding (from 1 to {MAX_DL_QUBITS} qubits). If ``None``,
                                          then it is set automatically from data size. Otherwise, it uses given number of qubits
                                          and automatically sets the minimal possible density. Data vector is extended
                                          with zero padding if necessary. The general scaling of the data size,
                                          which can be encoded, is O(``num_qubits`` * ``density`` ^2), up to small
                                          corrections. Defaults to ``None``.
                real (bool): if True, then a real quantum state is prepared, otherwise imaginary part is also used, doubling
                             the amount of features, which can be encoded in the same manifold. Defaults to ``True``.
                periodicity (bool): if True, then additional tangent transform is performed over data, adding periodicity
                                    properties to the encoding. With ``density==1`` it matches angular encoding.
                                    Defaults to ``False``.
                num_layers (int): The number of layers in the generated circuit (from 1 to {MAX_DL_LAYERS} layers).
                                  More layers can improve the quality of the output
                                  vector at the cost of a deeper circuit. Defaults to 2.
                truncation_cutoff (Real): The entanglement cutoff for later layers, from 0 to 1. Increasing this threshold may
                                          result in a smaller (but more approximate) circuit. Defaults to ``1e-6``.
                fine_tuning_iterations (int): The maximum number of fine-tuning iterations to perform after each layer is added.
                                              Increasing this limit may improve the quality of the circuit by using more classical
                                              resources. Defaults to 20, maximum is {MAX_DL_FINE_TUNING_ITERATIONS}.
                max_time (int | float): Soft time limit for the job (in seconds).
                                The data loading job will first always produce the initial result and then limit the fine-tuning
                                stage by the remaining time left. If time limit exceeds during the fine-tuning - the best
                                current result will be returned. Defaults to {MAX_DATA_LOADING_TIME}
                                ({MAX_DATA_LOADING_TIME // 60} min). Max allowed job time is {MAX_DATA_LOADING_TIME // 60} min.
                                The job can take more wall clock time than user specified `max_time` due to latency,
                                initialization overheads or if the initial result already takes more time.
                name (str | None): The name for the job and the produced circuit. If ``None`` (default), a name will be
                                   automatically generated.
        """

    @errors.graceful_api_errors_message
    @format_docstring(ARGS=_entangled_manifold_embedding_args, FEATURE_ORDER=_EME_FEATURE_ORDER_NOTE)
    def entangled_manifold_embedding(
        self,
        data: Sequence[Real],
        density: int | None = 2,
        num_qubits: int | None = None,
        real: bool = True,
        periodicity: bool = False,
        num_layers: int = 2,
        truncation_cutoff: Real = 1e-6,
        fine_tuning_iterations: int = 20,
        max_time: int | float = MAX_DATA_LOADING_TIME,
        name: str | None = None,
        job_description: str | None = None,
    ) -> DataLoadingJobModel:
        """Generate a quantum circuit that produces entangled manifold embedding of the real data into a quantum state of a
        controllable entanglement.
        The size of the Hilbert space, where the embedding is produced, is controlled by the `density` parameter.
        Using larger density results in usage of more entangled states for the embedding, which allows to encode more features,
        but results in more complicated quantum circuits.

        Given a vector of data, this method creates a Data Loading job that runs in the Haiqu cloud. The result of this job is a
        circuit which can be used to supply the vector to a quantum algorithm for processing.

        The complexity and quality of the generated circuit can be controlled by the ``num_layers``, ``truncation_cutoff``, and
        ``fine_tuning_iterations`` parameters.
        {FEATURE_ORDER}
        Args:{ARGS}
        Returns:
            DataLoadingJobModel: The Data Loading job that will generate the circuit for the data vector.
                Call ``job.result()`` to retrieve a Qiskit-compatible gate (``HaiquCircuitGate``) that performs the entangled
                manifold embedding of the input vector. ``job.quality`` is the achieved encoding fidelity vs. the ideal
                embedded state; ``job.info`` exposes loader metadata (``fidelity``).
                Run ``help(job.result)`` for the full description of result and ``info`` contents.

        Examples:
            >>> # loading a state with angular encoding
            >>> feature_vector = [1, 2, 3, 4, 5]
            >>> job = haiqu.entangled_manifold_embedding(data=feature_vector, density=1, periodicity=True, name="Angular")
            >>> ae_gate = job.result()  # ae_gate is a Qiskit-compatible gate
            >>> fidelity = job.quality
            >>> print(f"Angular encoding was loaded with fidelity {{fidelity:.6f}}")
            Angular encoding was loaded with fidelity 1.000000
            >>> print(f"Angular encoding required {{job.num_qubits}} qubits")
            Angular encoding required 5 qubits
            >>> circuit = qiskit.QuantumCircuit(job.num_qubits)
            >>> circuit.append(ae_gate, range(job.num_qubits))
            >>> circuit.draw()
                 ┌────────────────────────────────────────────────────────────┐
            q_0: ┤0                                                           ├
                 │                                                            │
            q_1: ┤1                                                           ├
                 │                                                            │
            q_2: ┤2 Haiqucircuit(circ-12345678-1234-5678-1234-567812345678,5) ├
                 │                                                            │
            q_3: ┤3                                                           ├
                 │                                                            │
            q_4: ┤4                                                           ├
                 └────────────────────────────────────────────────────────────┘

            >>> # loading a state into a more entangled Hilbert subspace
            >>> feature_vector = [1, 2, 3, 4, 5]
            >>> job = haiqu.entangled_manifold_embedding(data=feature_vector, density=2, name="EME")
            >>> eme_gate = job.result()  # eme_gate is a Qiskit-compatible gate
            >>> fidelity = job.quality
            >>> print(f"Entangled Manifold Embedding was loaded with fidelity {{fidelity:.6f}}")
            Entangled Manifold Embedding was loaded with fidelity 1.000000
            >>> print(f"Entangled Manifold Embedding required {{job.num_qubits}} qubits")
            Entangled Manifold Embedding required 3 qubits

        """
        self._check_experiment()
        name, parameters = self._prepare_entangled_manifold_embedding_params(
            data, density, real, periodicity, num_layers, truncation_cutoff, fine_tuning_iterations, max_time, name
        )

        return self._client.data_loading(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.ENTANGLED_MANIFOLD_EMBEDDING.value,
                name=name,
                description=job_description,
                num_qubits=num_qubits,
                experiment_id=self._experiment.id,
                parameters=parameters,
            )
        )

    @staticmethod
    def _prepare_mps_loading_params(
        mps,
        shape,
        num_layers,
        truncation_cutoff,
        fine_tuning_iterations,
        max_time,
        name,
    ):
        # Numeric limits are enforced by DataLoadingSubmitModel schema validation.
        parameters = {
            "mps": mps,
            "shape": shape,
            "num_layers": num_layers,
            "truncation_cutoff": truncation_cutoff,
            "fine_tuning_iterations": fine_tuning_iterations,
            "max_time": max_time,
        }

        if name is None:
            name = "MpsLoading()"

        return name, parameters

    _mps_loading_args = f"""
            mps (Sequence): The MPS in either standard or Vidal form. Standard form expects a list of rank-3
                             site tensors (one per each qubit). Vidal form is a tuple of site and bond tensors,
                             where bonds tensors are rank-1 or diagonal rank-2 tensors. The MPS type is determined automatically.
                             Standard form includes left- and right-canonical forms, while Vidal form includes
                             central canonical form. The maximal bond dimension we accept is {MAX_DL_MPS_BOND_DIMENSION}.
            shape (str): shape of site tensors of the MPS. Site tensors are rank-3 tensors. Shape defines
                         the order of axes in it.
                         p - physical index, l - left index, r - right index.
                         Defaults to "plr", which is standard order in Qiskit.
            num_layers (int): The number of layers in the generated circuit (from 1 to {MAX_DL_LAYERS} layers). More layers can
                              improve the quality of the output vector at the cost of a deeper circuit. Defaults to 2.
            truncation_cutoff (Real): The entanglement cutoff for later layers, from 0 to 1. Increasing this threshold may result
                                      in a smaller (but more approximate) circuit. Defaults to ``1e-6``.
            fine_tuning_iterations (int): The maximum number of fine-tuning iterations to perform after each layer is added.
                                          Increasing this limit may improve the quality of the circuit by using more classical
                                          resources. Defaults to 20, maximal is {MAX_DL_FINE_TUNING_ITERATIONS}.
            max_time (int | float): Soft time limit for the job (in seconds).
                            The data loading job will first always produce the initial result and then limit the fine-tuning
                            stage by the remaining time left. If time limit exceeds during the fine-tuning - the best
                            current result will be returned. Defaults to {MAX_DATA_LOADING_TIME}
                            ({MAX_DATA_LOADING_TIME // 60} min). Max allowed job time is {MAX_DATA_LOADING_TIME // 60} min.
                            The job can take more wall clock time than user specified `max_time` due to latency,
                            initialization overheads or if the initial result already takes more time.
            name (str | None): The name for the job and the produced circuit. If ``None`` (default), a name will be automatically
                               generated.
    """

    @errors.graceful_api_errors_message
    @format_docstring(ARGS=_mps_loading_args, MPS_ORDER=_MPS_QUBIT_ORDER_NOTE)
    def mps_loading(
        self,
        mps: Sequence,
        shape: str = "plr",
        num_layers: int = 2,
        truncation_cutoff: Real = 1e-6,
        fine_tuning_iterations: int = 20,
        max_time: int | float = MAX_DATA_LOADING_TIME,
        name: str | None = None,
        job_description: str | None = None,
    ) -> DataLoadingJobModel:
        """Generate a quantum circuit that prepares a quantum state from matrix product state (MPS).
        The MPS is normalized in the process, and expect physical index to be of size 2.

        Given a MPS, this method creates a Data Loading job that runs in the Haiqu cloud. The result of this job is a
        circuit which can be used to supply the state to a quantum algorithm for processing.

        Two MPS formats are supported:

        1. Standard form (only site tensors)::

            │     │     │     │        │
            A₁ ── A₂ ── A₃ ── A₄ ─ ⋯ ─ Aₙ

        2. Vidal form (site and bond tensors)::

            │           │           │              │
            Γ₁ ── Λ₁ ── Γ₂ ── Λ₂ ── Γ₃ ── Λ₃ ─ ⋯ ─ Γₙ

        The complexity and quality of the generated circuit can be controlled by the ``num_layers``, ``truncation_cutoff``, and
        ``fine_tuning_iterations`` parameters. Passing the MPS with high bond dimension may degrade the
        quality and synthesis time.
        {MPS_ORDER}
        Args:{ARGS}
        Returns:
            DataLoadingJobModel: The Data Loading job that will generate the circuit from the MPS.

        Examples:
            Loading Qiskit MPS:

            >>> qc = qiskit.QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)  # prepare test Bell state
            >>> qc.save_matrix_product_state(label="mps")
            >>> mps = AerSimulator().run(qc).result().data(0)["mps"]  # get MPS from Aer Simulator
            >>> job = haiqu.mps_loading(mps)
            >>> mps_gate = job.result()
            >>> print(f"MPS was loaded with fidelity {{job.fidelity:.3f}}")
            MPS was loaded with fidelity 1.000
            >>> mps_qc = qiskit.QuantumCircuit(2)
            >>> mps_qc.compose(mps_gate, inplace=True)
            >>> print(haiqu.statevector_run(mps_qc).result())  # confirm the Bell state was loaded
            [array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j])]

            Preparing a ground state of a Hamiltonian:

            >>> import quimb.tensor as qtn  # !pip install quimb (if not present)
            >>> H = qtn.MPO_ham_heis(4, j=1.0, cyclic=False)  # Heisenberg hamiltonian
            >>> dmrg = qtn.DMRG2(H, bond_dims=[4, 8])
            >>> dmrg.solve(tol=1e-12)  # find the ground state
            >>> job = haiqu.mps_loading(dmrg.state.arrays, shape="lpr")  # Quimb uses another shape
            >>> heis_gs = qiskit.QuantumCircuit(4)
            >>> heis_gs.compose(job.result(), inplace=True)  # circuit preparing the ground state
            >>> print(f"Ground state was loaded with fidelity {{job.fidelity:.3f}}")
            Ground state was loaded with fidelity 1.000
        """
        self._check_experiment()
        name, parameters = self._prepare_mps_loading_params(
            mps, shape, num_layers, truncation_cutoff, fine_tuning_iterations, max_time, name
        )

        return self._client.data_loading(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.MPS_LOADING.value,
                name=name,
                description=job_description,
                experiment_id=self._experiment.id,
                parameters=parameters,
            )
        )

    @staticmethod
    def _prepare_function_loading_params(
        num_qubits,
        func,
        interval_start,
        interval_end,
        num_layers,
        truncation_cutoff,
        fine_tuning_iterations,
        max_time,
        name,
    ):
        # Numeric limits are enforced by DataLoadingSubmitModel schema validation.
        # `func` is accepted as a SymPy expression or a string and normalized to a string for transport.
        # Only the single real variable `x` is allowed. The imaginary unit `I` is a SymPy constant, not a
        # free symbol, so complex-valued functions such as `exp(I*x)` pass the single-variable check.
        try:
            expression = sympy.sympify(func)
        except (sympy.SympifyError, SyntaxError, TypeError) as exc:
            raise ValueError(f"Could not parse `func` as a SymPy expression: {func!r}") from exc

        extra_symbols = {s for s in expression.free_symbols if s.name != "x"}
        if extra_symbols:
            raise ValueError(f"`func` must be an expression in the single variable `x`, but also got symbols {extra_symbols}")

        func_str = str(expression)

        parameters = {
            "func": func_str,
            "interval_start": interval_start,
            "interval_end": interval_end,
            "num_layers": num_layers,
            "truncation_cutoff": truncation_cutoff,
            "fine_tuning_iterations": fine_tuning_iterations,
            "max_time": max_time,
        }

        if name is None:
            name = f"FunctionLoading({func_str})"

        return name, parameters

    _function_loading_args = f"""
            num_qubits (int): The number of qubits in the generated circuit (from 1 to {MAX_DL_QUBITS} qubits).
            func (str | sympy.Expr): The function to encode, given as a SymPy expression or a string in the single
                                     variable ``x``.
            interval_start (Real): The beginning of the interval on which the function is sampled.
            interval_end (Real): The end of the interval on which the function is sampled. The interval width
                                 ``interval_end - interval_start`` must be at least {MIN_DL_INTERVAL_WIDTH}.
            num_layers (int): The number of layers in the generated circuit (from 1 to {MAX_DL_LAYERS} layers).
                              More layers can improve the quality of the output
                              function at the cost of a deeper circuit. Defaults to 2.
            truncation_cutoff (Real): The entanglement cutoff for later layers, from 0 to 1. Increasing this threshold may result
                                      in a smaller (but more approximate) circuit. Defaults to ``1e-6``.
            fine_tuning_iterations (int): The maximum number of fine-tuning iterations to perform after each layer is added.
                                          Increasing this limit may improve the quality of the circuit by using more classical
                                          resources. Defaults to 20, maximal is {MAX_DL_FINE_TUNING_ITERATIONS}.
            max_time (int | float): Soft time limit for the job (in seconds).
                            The data loading job will first always produce the initial result and then limit the fine-tuning
                            stage by the remaining time left. If time limit exceeds during the fine-tuning - the best
                            current result will be returned. Defaults to {MAX_DATA_LOADING_TIME}
                            ({MAX_DATA_LOADING_TIME // 60} min). Max allowed job time is {MAX_DATA_LOADING_TIME // 60} min.
                            The job can take more wall clock time than user specified `max_time` due to latency,
                            initialization overheads or if the initial result already takes more time.
            name (str | None): The name for the job and the produced circuit. If ``None`` (default), a name will be automatically
                               generated.
    """

    @errors.graceful_api_errors_message
    @format_docstring(
        ARGS=_function_loading_args,
        AMP_INDEXING=_AMPLITUDE_INDEXING_NOTE,
        DISCRETIZATION=_DISCRETIZATION_INDEXING_NOTE,
    )
    def function_loading(
        self,
        num_qubits: int,
        func: str | sympy.Expr,
        interval_start: Real,
        interval_end: Real,
        num_layers: int = 2,
        truncation_cutoff: Real = 1e-6,
        fine_tuning_iterations: int = 20,
        max_time: int | float = MAX_DATA_LOADING_TIME,
        name: str | None = None,
    ) -> DataLoadingJobModel:
        """Generate a quantum circuit that prepares the values of a single-variable function in its amplitudes.

        Given a function ``f(x)``, this method creates a Data Loading job that runs in the Haiqu cloud. The result of this
        job is a circuit gate which prepares a state whose amplitudes are the function values, L2-normalized as a quantum state.
        The resulting gate can be used to supply the function to a quantum algorithm for processing.

        The function is provided as a `SymPy <https://www.sympy.org>`_ expression or as a string, and must depend on the
        single variable ``x`` only (e.g. ``"sin(x)"``, ``"exp(-x**2)"``, ``"x**2 + 1"``). The variable ``x`` is real-valued
        and ranges over the real interval ``[interval_start, interval_end]``. The function values may be complex even though
        ``x`` is real; the imaginary unit can be written Python-style as ``1j`` or SymPy-style as ``I``
        (e.g. ``"exp(1j*x)"`` or ``"exp(I*x)"``).

        The function is discretized on a grid of ``2**num_qubits`` points: it is evaluated at the center (midpoint) of each
        bin of the interval, and the resulting values are normalized as a quantum state. Singularities and non-finite values
        (``nan``, ``+/-inf``) are nullified (set to 0).

        The complexity and quality of the generated circuit can be controlled by the ``num_layers``, ``truncation_cutoff``,
        and ``fine_tuning_iterations`` parameters.
        {AMP_INDEXING}
        {DISCRETIZATION}
        Args:{ARGS}
        Returns:
            DataLoadingJobModel: The Data Loading job that will generate the circuit for the function.
                Call ``job.result()`` to retrieve a Qiskit-compatible gate (``HaiquCircuitGate``) that prepares the function
                values. ``job.quality`` is the achieved state fidelity vs. the ideal target function; ``job.info`` exposes
                loader metadata (``fidelity``).
                Run ``help(job.result)`` for the full description of result and ``info`` contents.

        Examples:
            Encoding a Gaussian given as a SymPy expression:

            >>> import sympy
            >>> x = sympy.Symbol("x")
            >>> job = haiqu.function_loading(num_qubits=6, func=sympy.exp(-x**2), interval_start=-3, interval_end=3)
            >>> fl_gate = job.result()  # fl_gate is a Qiskit-compatible gate
            >>> print(f"Function was loaded with fidelity {{job.quality:.6f}}")
            Function was loaded with fidelity 0.999518

            The same function given as a string:

            >>> job = haiqu.function_loading(num_qubits=6, func="exp(-x**2)", interval_start=-3, interval_end=3)

            Encoding a sine wave:

            >>> job = haiqu.function_loading(num_qubits=6, func="sin(x)", interval_start=-5, interval_end=5)

            Encoding a complex-valued function (``1j`` and ``I`` are equivalent):

            >>> job = haiqu.function_loading(num_qubits=6, func="exp(1j*x)", interval_start=0, interval_end=10)
        """
        self._check_experiment()
        name, parameters = self._prepare_function_loading_params(
            num_qubits,
            func,
            interval_start,
            interval_end,
            num_layers,
            truncation_cutoff,
            fine_tuning_iterations,
            max_time,
            name,
        )

        return self._client.data_loading(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.FUNCTION_LOADING.value,
                name=name,
                num_qubits=num_qubits,
                experiment_id=self._experiment.id,
                parameters=parameters,
            )
        )

    @staticmethod
    def _prepare_fourier_loading_params(
        fourier_coefficients,
        num_qubits,
        min_freqs,
        long_range,
        num_layers,
        truncation_cutoff,
        fine_tuning_iterations,
        max_time,
        name,
    ):
        # Numeric limits (num_layers, truncation_cutoff, fine_tuning_iterations, max_time) are enforced
        # by DataLoadingSubmitModel schema validation, which also coerces them to serializable types.
        if not isinstance(long_range, bool):
            raise ValueError("long_range must be a boolean")

        try:
            coefficients = np.asarray(fourier_coefficients, dtype=complex)
        except (TypeError, ValueError):
            raise ValueError("fourier_coefficients must form a rectangular array of numbers") from None
        if coefficients.ndim < 1 or coefficients.size == 0:
            raise ValueError("fourier_coefficients must be a rectangular array with at least one dimension and one entry")
        if not np.isfinite(coefficients).all():
            raise ValueError("fourier_coefficients must be finite (no nan or inf)")
        if np.linalg.norm(coefficients) == 0:
            raise ValueError("fourier_coefficients must have a non-zero norm")

        num_dimensions = coefficients.ndim

        def normalize_per_dimension(name: str, values: Any) -> list[int] | None:
            """Normalize a per-dimension integer argument to one int per dimension, or None.

            A scalar is broadcast to every dimension and a sequence is taken as one value per dimension.
            """
            if values is None:
                return None
            if isinstance(values, (int, np.integer)):
                return [int(values)] * num_dimensions
            if isinstance(values, str) or not isinstance(values, (Sequence, np.ndarray)):
                raise ValueError(
                    f"{name} must be an integer, a sequence of integers or None, but {type(values).__name__} was given"
                )
            if isinstance(values, np.ndarray) and values.ndim != 1:
                raise ValueError(f"{name} must be an integer or a one-dimensional sequence of integers")
            if len(values) != num_dimensions:
                raise ValueError(
                    f"{name} has {len(values)} entries, which does not match the number of dimensions {num_dimensions}"
                )
            if not all(isinstance(value, (int, np.integer)) for value in values):
                raise ValueError(f"Each entry of {name} must be an integer")
            return [int(value) for value in values]

        min_freqs = normalize_per_dimension("min_freqs", min_freqs)
        num_qubits = normalize_per_dimension("num_qubits", num_qubits)

        minimum_qubits = [int(np.ceil(np.log2(size))) for size in coefficients.shape]
        if num_qubits is not None:
            # Every dimension needs enough qubits to hold its frequencies; the extra ones interpolate.
            for dimension, (requested, minimum) in enumerate(zip(num_qubits, minimum_qubits)):
                if requested < minimum:
                    raise ValueError(
                        f"num_qubits[{dimension}]={requested} is too small to hold the {coefficients.shape[dimension]} "
                        f"frequencies of dimension {dimension}, which need at least {minimum} qubits"
                    )

        # A dimension holding a single frequency needs no qubits of its own, so an array that is length
        # one along every axis resolves to an empty register unless interpolation qubits are requested.
        if sum(num_qubits if num_qubits is not None else minimum_qubits) < 1:
            raise ValueError(
                "The circuit would have no qubits, because every dimension of fourier_coefficients holds a single "
                "frequency; pass num_qubits summing to at least one qubit to interpolate onto a non-trivial grid"
            )

        parameters = {
            "fourier_coefficients": coefficients,
            "num_qubits": num_qubits,
            "min_freqs": min_freqs,
            "long_range": long_range,
            "num_layers": num_layers,
            "truncation_cutoff": truncation_cutoff,
            "fine_tuning_iterations": fine_tuning_iterations,
            "max_time": max_time,
        }

        if name is None:
            name = f"FourierLoading(shape:{tuple(coefficients.shape)})"

        return name, parameters

    _fourier_loading_args = f"""
            fourier_coefficients (Sequence[Number] | np.ndarray): The multidimensional array of Fourier coefficients,
                                     ascending in frequency along every axis, which is not the order ``np.fft.fftn``
                                     returns (see the frequency-layout note above). May be real or complex, and is
                                     normalized in the process. This is the layout that
                                     :func:`~haiqu.sdk.utils.compute_fourier_coefficients` returns, which is the recommended way
                                     to obtain it; an analytically known spectrum can equally be written down directly.
            num_qubits (int | Sequence[int] | None): The number of output qubits per dimension. An integer is shared by all
                                     dimensions, a sequence sets them individually in the order of the coefficient array's
                                     axes. Each entry must be at least ``ceil(log2(n))`` for a dimension holding ``n``
                                     frequencies; every qubit above that minimum interpolates the state onto a finer grid.
                                     If ``None`` (default), the minimum that fits the coefficients is used, which performs
                                     no interpolation.
            min_freqs (int | Sequence[int] | None): The integer frequency that the first coefficient of each dimension
                                     carries. An integer is shared by all dimensions, a sequence sets them individually
                                     in the order of the coefficient array's axes. If ``None`` (default), each dimension
                                     is assumed to hold a band centred on zero and takes ``-(n // 2)`` for its length
                                     ``n``. It is applied as a frequency-shift layer, so a value that disagrees with how
                                     the coefficients were obtained silently encodes the wrong state — see the
                                     frequency-layout note above.
            long_range (bool): Whether to allow long-range gates in the Fourier part of the circuit. The prepared state is
                               the same either way, while ``False`` (default) keeps the gates on a linear chain.
            num_layers (int): The number of layers used to synthesize the coefficients (from 1 to {MAX_DL_LAYERS} layers).
                              More layers can improve the quality of the output at the cost of a deeper circuit.
                              Defaults to 2.
            truncation_cutoff (Real): The entanglement cutoff for later layers. Increasing this threshold may result in a
                                      smaller (but more approximate) circuit. Defaults to ``1e-6``.
            fine_tuning_iterations (int): The maximum number of fine-tuning iterations to perform after each layer is added.
                                          Increasing this limit may improve the quality of the circuit by using more
                                          classical resources. Defaults to 20, maximal is {MAX_DL_FINE_TUNING_ITERATIONS}.
            max_time (int | float): Soft time limit for the job (in seconds).
                            The data loading job will first always produce the initial result and then limit the fine-tuning
                            stage by the remaining time left. If time limit exceeds during the fine-tuning - the best
                            current result will be returned. Defaults to {MAX_DATA_LOADING_TIME}
                            ({MAX_DATA_LOADING_TIME // 60} min). Max allowed job time is {MAX_DATA_LOADING_TIME // 60} min.
                            The job can take more wall clock time than user specified `max_time` due to latency,
                            initialization overheads or if the initial result already takes more time.
            name (str | None): The name for the job and the produced circuit. If ``None`` (default), a name will be
                               automatically generated.
            job_description (str | None): The description for the job.
    """

    @errors.graceful_api_errors_message
    @format_docstring(
        ARGS=_fourier_loading_args, FOURIER_FREQUENCY=_FOURIER_FREQUENCY_NOTE, FOURIER_INDEXING=_FOURIER_INDEXING_NOTE
    )
    def fourier_loading(
        self,
        fourier_coefficients: Sequence[Number] | np.ndarray,
        num_qubits: int | Sequence[int] | None = None,
        min_freqs: int | Sequence[int] | None = None,
        long_range: bool = False,
        num_layers: int = 2,
        truncation_cutoff: Real = 1e-6,
        fine_tuning_iterations: int = 20,
        max_time: int | float = MAX_DATA_LOADING_TIME,
        name: str | None = None,
        job_description: str | None = None,
    ) -> DataLoadingJobModel:
        """Generate a quantum circuit that prepares a band-limited multidimensional state from its Fourier coefficients.

        Given the Fourier coefficients of a state, this method creates a Data Loading job that runs in the Haiqu cloud. The
        coefficients are synthesized on the smallest register that fits them, and an optimized quantum Fourier transform per
        dimension expands them onto the output grid. The result of this job is a circuit which can be used to supply the
        state to a quantum algorithm for processing. It pays off only for spectra concentrated in few frequencies, so the
        state is assumed to represent a periodic function; symmetrizing a dimension can make a non-periodic target more
        periodic.

        The coefficients are normally obtained with :func:`~haiqu.sdk.utils.compute_fourier_coefficients`, which takes an array of
        values on a uniform grid and returns ``fourier_coefficients`` and ``min_freqs`` in exactly the form this method expects.

        The complexity and quality of the generated circuit can be controlled by the ``num_layers``,
        ``truncation_cutoff``, and ``fine_tuning_iterations`` parameters.
        {FOURIER_FREQUENCY}
        {FOURIER_INDEXING}
        .. note::

            **``job.quality`` excludes the truncation error.** It is the fidelity of synthesizing the coefficients you
            supplied, which are taken as the exact target, so how they were obtained does not enter it. Discarding
            frequencies is a separate, purely classical approximation, measured by the fidelity that
            :func:`~haiqu.sdk.utils.compute_fourier_coefficients` returns. The fidelity of the final state against your original
            array is approximately the product of the two.

        Args:{ARGS}
        Returns:
            DataLoadingJobModel: The Data Loading job that will generate the circuit for the band-limited state.
                Call ``job.result()`` to retrieve a Qiskit-compatible gate (``HaiquCircuitGate``) that prepares the state.
                ``job.quality`` is the achieved state fidelity vs. the supplied coefficients; ``job.info`` exposes loader
                metadata (``fidelity``, ``num_qubits_per_dimension``).
                Run ``help(job.result)`` for the full description of result and ``info`` contents.

        Examples:
            An exact finite Fourier series. ``cos(2*pi*3*x)`` has only the -3 and +3 harmonics, so retaining the band
            ``(-3, 3)`` discards nothing and both fidelities are 1:

            >>> import numpy as np
            >>> from haiqu.sdk.utils import compute_fourier_coefficients
            >>> grid = np.arange(64) / 64
            >>> cosine = np.cos(2 * np.pi * 3 * grid)
            >>> coefficients, min_freqs, truncation_fidelity = compute_fourier_coefficients(
            ...     cosine, freq_ranges=[(-3, 3)], return_fidelity=True
            ... )
            >>> print(coefficients.shape, min_freqs, f"{{truncation_fidelity:.6f}}")
            (7,) [-3] 1.000000
            >>> job = haiqu.fourier_loading(coefficients, num_qubits=6, min_freqs=min_freqs)
            >>> fl_gate = job.result()  # fl_gate is a Qiskit-compatible gate
            >>> print(f"Cosine was loaded with fidelity {{job.quality:.6f}}")
            Cosine was loaded with fidelity 1.000000

            A 2D Gaussian sampled on a 32x32 grid, encoded onto a 256x256 grid. Only 81 coefficients are sent, and the
            eight output qubits per dimension interpolate the rest:

            >>> axis = np.arange(32) / 32
            >>> x, y = np.meshgrid(axis, axis, indexing="ij")
            >>> gaussian = np.exp(-((x - 0.4) ** 2 + (y - 0.5) ** 2) / (4 * 0.15 ** 2))
            >>> coefficients, min_freqs, truncation_fidelity = compute_fourier_coefficients(
            ...     gaussian, freq_ranges=[(-4, 4), (-4, 4)], return_fidelity=True
            ... )
            >>> print(coefficients.shape, min_freqs, f"{{truncation_fidelity:.6f}}")
            (9, 9) [-4, -4] 0.999535
            >>> job = haiqu.fourier_loading(coefficients, num_qubits=8, min_freqs=min_freqs)
            >>> fl_gate = job.result()
            >>> print(f"{{job.num_qubits}} qubits, synthesis fidelity {{job.quality:.6f}}")
            16 qubits, synthesis fidelity 0.999998
            >>> print(f"Fidelity against the original array is about {{truncation_fidelity * job.quality:.6f}}")
            Fidelity against the original array is about 0.999533
        """
        self._check_experiment()
        name, parameters = self._prepare_fourier_loading_params(
            fourier_coefficients,
            num_qubits,
            min_freqs,
            long_range,
            num_layers,
            truncation_cutoff,
            fine_tuning_iterations,
            max_time,
            name,
        )

        # The resolved total width is left to the backend, as in `mps_loading`: `num_qubits` here is
        # per dimension, while the top-level field is a single count.
        return self._client.data_loading(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.FOURIER_LOADING.value,
                name=name,
                description=job_description,
                experiment_id=self._experiment.id,
                parameters=parameters,
            )
        )

    # TODO: accept CompressionOptions instead of raw string params (compression_level,
    # noise_profile, fine_tuning, approximation_level). Both state_compression and
    # state_compression_2d route through here, so a single change covers both.
    def _prepare_state_compression_params(
        self,
        circuit: QuantumCircuit | CircuitModel = None,  # Deprecated
        circuits: list[QuantumCircuit] | list[CircuitModel] = None,
        compression_level: str = "balanced",
        noise_profile: str = "default",
        fine_tuning: str = "low",
        max_time: int | float = MAX_COMPRESSION_TIME,
        approximation_level: int | None = None,
        device_id: str | None = None,
    ):
        # Numeric limits (including max_time) are enforced by StateCompressionParams schema validation.
        if circuit is not None and circuits is not None:
            raise ValueError("Only one of `circuit` and `circuits` can be specified.")
        if circuit is not None:
            circuits = [circuit]
            warnings.warn("The 'circuit' parameter is deprecated; use 'circuits' instead.", DeprecationWarning, stacklevel=2)

        logged_circuits = self._prepare_circuits(circuits)

        parameters = {
            "compression_level": compression_level,
            "noise_profile": noise_profile,
            "fine_tuning": fine_tuning,
            "approximation_level": approximation_level,
            "max_time": max_time,
        }
        if device_id is not None:
            parameters["device_id"] = device_id

        return parameters, logged_circuits

    @errors.graceful_api_errors_message
    @format_docstring(
        MAX_TIME=MAX_COMPRESSION_TIME,
        MAX_TIME_MIN=MAX_COMPRESSION_TIME // 60,
        MAX_QUBITS=MAX_COMPRESSION_QUBITS,
        MIN_APPROX=MIN_COMPRESSION_APPROXIMATION_LEVEL,
        MAX_APPROX=MAX_COMPRESSION_APPROXIMATION_LEVEL,
    )
    def state_compression(
        self,
        circuit: QuantumCircuit | CircuitModel = None,
        circuits: list[QuantumCircuit] | list[CircuitModel] = None,
        compression_level: str = "balanced",
        noise_profile: str = "default",
        fine_tuning: str = "low",
        max_time: int | float = MAX_COMPRESSION_TIME,
        approximation_level: int | None = None,
    ) -> StateCompressionJobModel | list[StateCompressionJobModel]:
        """Compress an arbitrary quantum circuit.

        Haiqu's state compression is an approximate fixed-input-state compilation method to extend the effective depth of
        circuits that can be executed on noisy hardware. It features several tunable parameters to adjust the trade-off
        between compression level and circuit quality, allowing the user to tailor the compression to the circuit and
        device noise characteristics.

        Both the input and output circuits are assumed to be applied to the all-zero state (`|00⋯0⟩`). The action of the circuit
        on other input states is not preserved by the compression.

        .. note::

            * ``job.result()`` returns a ``CircuitModel``, not a plain ``QuantumCircuit`` or
              :class:`~haiqu.sdk.gates.HaiquCircuitGate`. The model identifies a cloud-stored circuit by
              its ID. Pass ``result`` directly to :meth:`transpile` or :meth:`run`, call ``result.to_gate()`` to embed
              it in a larger Qiskit circuit, or persist ``result.id`` if you need to retrieve it later
              with :meth:`get_circuit`. Metrics are on ``result.analytics`` (for example
              ``analytics.depth``, ``analytics.gates_2q``) or ``result.core_metrics()``.
            * ``job.quality`` is a noiseless compilation-quality metric, not a bound on observable accuracy —
              validate the observables you care about rather than treating ``quality`` as an accuracy
              surrogate.
            * ``compression_level`` and ``approximation_level`` are independent knobs: the former selects the compression target
              (the fraction of the input circuit that is compressed), and the latter affects the complexity of the output circuit.
              Setting ``approximation_level`` explicitly replaces the automatic selection derived from ``noise_profile``; it does
              not change what ``compression_level`` controls.

        Args:
            circuit (QuantumCircuit | CircuitModel): Deprecated. The quantum circuit to be compressed.
                                                     Circuit must have no more than {MAX_QUBITS} qubits.
            circuits (list[QuantumCircuit] | list[CircuitModel]): The quantum circuit(s) to be compressed.
            compression_level (str): The qualitative compression level. Increased compression level will lead to
                                     larger part of the input circuit being compressed.
                                     Four options are available:

                                     * "low": best used for shallow input circuits or very low noise levels
                                     * "balanced" (default): gives the best performance for most circuits and noise profiles
                                     * "high": may sometimes yield better results for very deep circuits
                                     * "max": the largest possible part of the input circuit will be compressed,
                                              yielding the most extreme depth reduction. Recommended to combine
                                              with custom approximation level to tune the quality.

            noise_profile (str): The device noise profile to assume during compression. The currently available options are:
                                 "ibm_eagle_r3", "ibm_heron_r1", "ibm_heron_r2" (default), "ibm_heron_r3",
                                 "iqm_garnet" and "iqm_emerald". Used to automatically set the approximation level.

            fine_tuning (str): The extent to which classical resources should be used to further improve the compressed circuit.
                               Three options are available:

                               * "disabled": no fine-tuning is performed, yielding the lowest latency
                               * "low" (default): best balance between speed and accuracy
                               * "heavy": improved circuit accuracy, but time-intensive

            max_time (int | float): Soft time limit for the job (in seconds).
                            The compression job will first always produce the initial result and then limit the fine-tuning
                            stage by the remaining time left. If time limit exceeds during the fine-tuning - the best
                            current result will be returned. Defaults to {MAX_TIME}
                            ({MAX_TIME_MIN} min). Max allowed job time is {MAX_TIME_MIN} min.
                            The job can take more wall clock time than user specified `max_time` due to latency,
                            initialization overheads or if the initial result already takes more time.
            approximation_level (int | None): A small integer related to circuit complexity. Larger values improve the noiseless
                                              quality metric, but may degrade noisy performance. Defaults to ``None``, which
                                              corresponds to auto-selection using the chosen ``noise_profile``. Can be set from
                                              {MIN_APPROX} (very weak approximation) to {MAX_APPROX} (very high approximation).
                                              Larger approximation level values lead to slower fine-tuning. For majority of
                                              applications recommended values are generally ranged from 1 to 5.

        Returns:
            StateCompressionJobModel | list[StateCompressionJobModel]: The State Compression job(s) that will generate the
                compressed circuit(s).
                Call ``job.result()`` to retrieve the compressed circuit as a ``CircuitModel``. ``job.quality`` is the
                compression quality, computed in a noiseless setting; ``job.info`` exposes compression metadata
                (``compression_quality``, ``success``, ``compression_status``, ``compression_time``,
                ``compression_percent``, ``approximation_level``).
                Use ``job.progress()`` for live status updates and ``help(job.result)`` for the full description of result
                and ``info`` contents.

        Examples:
            Generate a circuit:

            >>> from qiskit.circuit.random import random_circuit
            >>> qc = random_circuit(num_qubits=50, depth=5, max_operands=4, seed=2025, measure=False)
            >>> circuit_aer = haiqu.transpile(qc, device=haiqu.get_device("aer_simulator"), basis_gates=["cx", "u3"])
            >>> print(f"{{circuit_aer.analytics.gates_2q}} two-qubit gates in the original circuit")
            278 two-qubit gates in the original circuit

            Submit a State Compression job to shrink it:

            >>> job = haiqu.state_compression(qc)
            >>> circuit_comp = job.result()
            >>> quality = job.quality
            >>> print(f"Circuit is compressed with quality {{quality:.6f}}")
            Circuit is compressed with quality 0.898719

            Submit an Analytics job to confirm that the compressed circuit has far fewer two-qubit gates:

            >>> circuit_comp_aer = haiqu.transpile(circuit_comp, device=haiqu.get_device("aer_simulator"),
            ...                                    basis_gates=["cx", "u3"])
            >>> print(f"{{circuit_comp_aer.analytics.gates_2q}} two-qubit gates in the compressed circuit")
            95 two-qubit gates in the compressed circuit

            Batch submission of the State Compression jobs:

            >>> circuits = [random_circuit(num_qubits=20, depth=10, max_operands=4, seed=s, measure=False) for s in range(3)]
            >>> jobs = haiqu.state_compression(circuits=circuits)
            >>> for job in jobs:
            ...     circuit_comp = job.result()
            ...     quality = job.quality
            ...     print(f"Circuit is compressed with quality {{quality:.6f}}")
        """
        self._check_experiment()

        parameters, logged_circuits = self._prepare_state_compression_params(
            circuit=circuit,
            circuits=circuits,
            compression_level=compression_level,
            noise_profile=noise_profile,
            fine_tuning=fine_tuning,
            max_time=max_time,
            approximation_level=approximation_level,
        )

        jobs = self._client.compression(
            data=StateCompressionSubmitModel(
                experiment_id=self._experiment.id,
                circuit_ids=[c.id for c in logged_circuits],
                parameters=parameters,
                compression_type=CompressionJobType.STATE_COMPRESSION.value,
            )
        )
        if circuit is not None:
            return jobs[0]
        return jobs

    @errors.graceful_api_errors_message
    @format_docstring(
        MAX_TIME=MAX_COMPRESSION_TIME,
        MAX_TIME_MIN=MAX_COMPRESSION_TIME // 60,
        MIN_APPROX=MIN_COMPRESSION_APPROXIMATION_LEVEL,
        MAX_APPROX=MAX_COMPRESSION_APPROXIMATION_LEVEL,
    )
    def state_compression_2d(
        self,
        circuit: QuantumCircuit | CircuitModel = None,
        circuits: list[QuantumCircuit] | list[CircuitModel] = None,
        device: DeviceModel | None = None,
        device_id: str | None = None,
        compression_level: str = "balanced",
        noise_profile: str | None = None,
        fine_tuning: str = "disabled",
        max_time: int | float = MAX_COMPRESSION_TIME,
        approximation_level: int | None = None,
    ) -> StateCompressionJobModel | list[StateCompressionJobModel]:
        """Compress an arbitrary quantum circuit on a targeted device.
        2D state compression follows the topology of a device and produces an already
        transpiled circuit.

        Note:
            2D state compression is currently limited to heavy hex devices.

        Haiqu's 2D state compression is an approximate fixed-input-state compilation method to extend
        the effective depth of circuits that can be executed on noisy hardware.
        It features several tunable parameters to adjust the trade-off between compression
        level and circuit quality, allowing the user to tailor the compression to the circuit and device noise characteristics.

        Both the input and output circuits are assumed to be applied to the all-zero state (``|00⋯0⟩``). The action of the circuit
        on other input states is not preserved by the compression.

        Args:
            circuit (QuantumCircuit | CircuitModel): Deprecated. The quantum circuit to be compressed.
                                                     The circuit size must not exceed device's size.
            circuits (list[QuantumCircuit] | list[CircuitModel]): The quantum circuit(s) to be compressed.
            device (DeviceModel | None): The target device for compression. If specified, ``device_id`` is ignored.
            device_id (str | None): The ID of the target device for compression. Defaults to ``None``.
            compression_level (str): The qualitative compression level. Increased compression level will lead to
                                     larger part of the input circuit being compressed.
                                     Four options are available:

                                     * "low": best used for shallow input circuits or very low noise levels
                                     * "balanced" (default): gives the best performance for most circuits and noise profiles
                                     * "high": may sometimes yield better results for very deep circuits
                                     * "max": the largest possible part of the input circuit will be compressed,
                                              yielding the most extreme depth reduction. Recommended to combine
                                              with custom approximation level to tune the quality.

            noise_profile (str | None): The device noise profile to use during compression. See `state_compression` options.
                                        By default (None) the noise profile is automatically chosen to match the device.
                                        Used to automatically set the approximation level.

            fine_tuning (str): The extent to which classical resources should be used to further improve the compressed circuit.
                               Three options are available:

                               * "disabled" (default): no fine-tuning is performed, yielding the lowest latency
                               * "low": best balance between speed and accuracy
                               * "heavy": improved circuit accuracy, but time-intensive

            max_time (int | float): Soft time limit for the job (in seconds).
                            The compression job will first always produce the initial result and then limit the fine-tuning
                            stage by the remaining time left. If time limit exceeds during the fine-tuning - the best
                            current result will be returned. Defaults to {MAX_TIME}
                            ({MAX_TIME_MIN} min). Max allowed job time is {MAX_TIME_MIN} min.
                            The job can take more wall clock time than user specified `max_time` due to latency,
                            initialization overheads or if the initial result already takes more time.
            approximation_level (int | None): A small integer related to circuit complexity. Larger values improve the noiseless
                                              quality metric, but may degrade noisy performance. Defaults to ``None``, which
                                              corresponds to auto-selection using the chosen ``noise_profile``. Can be set from
                                              {MIN_APPROX} (very weak approximation) to {MAX_APPROX} (very high approximation).
                                              Larger approximation level values lead to slower fine-tuning. For majority of
                                              applications recommended values are generally ranged from 1 to 5.

        Returns:
            StateCompressionJobModel | list[StateCompressionJobModel]: The State Compression job(s) that will generate the
                compressed circuit(s).
                Call ``job.result()`` to retrieve the compressed circuit as a ``CircuitModel``, already transpiled to the
                target device topology. ``job.quality`` is the compression quality, computed in a noiseless setting;
                ``job.info`` exposes compression metadata (``compression_quality``, ``success``, ``compression_status``,
                ``compression_time``, ``compression_percent``, ``approximation_level``).
                Use ``job.progress()`` for live status updates and ``help(job.result)`` for the full description of result
                and ``info`` contents.

        Examples:
            Generate a circuit:

            >>> from qiskit.circuit.random import random_circuit
            >>> quantum_device = "fake_fez"
            >>> qc = random_circuit(num_qubits=50, depth=5, max_operands=4, seed=2025, measure=False)
            >>> circuit_fez = haiqu.transpile(qc, device=haiqu.get_device(quantum_device))
            >>> print(f"{{circuit_fez.analytics.gates_2q}} two-qubit gates in the circuit transpiled to a device")
            1125 two-qubit gates in the circuit transpiled to a device

            Submit a 2D State Compression job to shrink it:

            >>> job = haiqu.state_compression_2d(qc, device_id=quantum_device)
            >>> circuit_comp = job.result()
            >>> quality = job.quality
            >>> print(f"Circuit is compressed with quality {{quality:.6f}}")
            Circuit is compressed with quality 0.950118

            Check the analytics to compare amount of two-qubit gates on a device.
            Note that it is already transpiled to a device chosen in the compression call.

            >>> print(f"{{circuit_comp.analytics.gates_2q}} two-qubit gates in the compressed circuit")
            66 two-qubit gates in the compressed circuit

            Batch submission of the 2D State Compression jobs:

            >>> circuits = [random_circuit(num_qubits=20, depth=10, max_operands=4, seed=s, measure=False) for s in range(3)]
            >>> jobs = haiqu.state_compression_2d(circuits=circuits, device_id=quantum_device)
            >>> for job in jobs:
            ...     circuit_comp = job.result()
            ...     quality = job.quality
            ...     print(f"Circuit is compressed with quality {{quality:.6f}}")
        """
        self._check_experiment()

        # Validate device ID
        if device is None:
            if device_id is None:
                raise ValueError("The `device` or `device_id` is required for 2D compression.")
            else:
                # Validate that device exists
                self.get_device(device_id=device_id)

        # If both device and device_id are provided, device_id is ignored
        if device is not None:
            device_id = device.id

        parameters, logged_circuits = self._prepare_state_compression_params(
            circuit=circuit,
            circuits=circuits,
            compression_level=compression_level,
            noise_profile=noise_profile,
            fine_tuning=fine_tuning,
            max_time=max_time,
            approximation_level=approximation_level,
            device_id=device_id,
        )

        jobs = self._client.compression(
            data=StateCompressionSubmitModel(
                experiment_id=self._experiment.id,
                circuit_ids=[c.id for c in logged_circuits],
                parameters=parameters,
                compression_type=CompressionJobType.STATE_COMPRESSION_2D.value,
            )
        )
        if circuit is not None:
            return jobs[0]
        return jobs

    @errors.graceful_api_errors_message
    def su2_equivariant_compilation(
        self,
        target: QuantumCircuit | Gate | np.ndarray,
        *,
        target_fidelity: float = 0.99,
        max_layers: int = 6,
        num_restarts: int = 10,
        seed: int = 0,
    ) -> Su2EquivariantCompilationJobModel:
        """Compress an SU(2)-equivariant gate into a brick of 2-qubit ``su2`` gates.

        Submits a job that fits a shallow brickwork of 2-qubit ``su2`` gates to
        ``target`` and returns the compressed circuit. The headline use is
        compressing the exact 3-qubit equivariant gate into a 2q brick, which
        transpiles to substantially fewer two-qubit gates.

        The target must be SU(2)-equivariant (commute with the global spin
        generators); non-equivariant inputs fail the job. Inherently small-n:
        the fit needs the dense ``2^n`` by ``2^n`` target unitary, so the
        target is capped at 10 qubits. For larger systems, build a
        parametrized :func:`~haiqu.sdk.qml.su2_equivariant_ansatz` and optimise
        at the state level instead.

        Args:
            target (QuantumCircuit | Gate | np.ndarray): SU(2)-equivariant
                target. Accepts a ``QuantumCircuit``, a ``Gate``, or a
                ``2^n`` by ``2^n`` ``numpy.ndarray``.
            target_fidelity (float): Requested process fidelity to the target
                unitary (default ``0.99``); a gate (process) fidelity, distinct
                from the state fidelity used elsewhere in the SDK. This is the
                goal the fit aims for, not a guarantee: the returned circuit may
                fall short, so check ``job.fidelity`` on the result. The fit
                escalates brickwork depth trying to clear this bar.
            max_layers (int): Cap on brickwork depth before giving up.
            num_restarts (int): Number of optimiser restarts.
            seed (int): Random seed for restart initialisation.

        Returns:
            Su2EquivariantCompilationJobModel: The compression job. Call
            ``job.result()`` to retrieve the compressed circuit as a
            ``CircuitModel`` and read ``job.fidelity`` for the achieved
            process fidelity.

        Raises:
            ValueError: If ``target`` exceeds the 10-qubit limit.
            TypeError: If ``target`` is not a QuantumCircuit, Gate, or ndarray.

        Example:
            >>> from haiqu.sdk.qml import su2_equivariant_3_qubit_gate
            >>> target = su2_equivariant_3_qubit_gate(0.8, 1.2, 0.5, 2.1)
            >>> job = haiqu.su2_equivariant_compilation(target, target_fidelity=0.99)
            >>> circuit = job.result()
            >>> job.fidelity >= 0.99
            True

            Transpiling both to a device basis shows the two-qubit-gate drop:

            >>> dev = haiqu.get_device("fake_marrakesh")
            >>> orig = haiqu.transpile(target, device=dev)
            >>> comp = haiqu.transpile(circuit, device=dev)
            >>> comp.analytics.gates_2q < orig.analytics.gates_2q
            True
        """
        if isinstance(target, QuantumCircuit):
            qc = target
        elif isinstance(target, Gate):
            qc = QuantumCircuit(target.num_qubits)
            qc.append(target, list(range(target.num_qubits)))
        elif isinstance(target, np.ndarray):
            unitary = UnitaryGate(target)
            qc = QuantumCircuit(unitary.num_qubits)
            qc.append(unitary, list(range(unitary.num_qubits)))
        else:
            raise TypeError("target must be a QuantumCircuit, Gate, or numpy.ndarray; " f"got {type(target).__name__}")

        if qc.num_qubits > constants.MAX_SU2_EQUIVARIANT_COMPILATION_QUBITS:
            raise ValueError(
                f"su2_equivariant_compilation supports at most {constants.MAX_SU2_EQUIVARIANT_COMPILATION_QUBITS} qubits "
                f"(the fit builds a dense 2^n x 2^n unitary); got {qc.num_qubits}."
            )

        self._check_experiment()

        logged_circuit = self._get_or_create_circuit(circuit=qc)
        jobs = self._client.su2_equivariant_compilation(
            data=StateCompressionSubmitModel(
                experiment_id=self._experiment.id,
                circuit_ids=[logged_circuit.id],
                parameters={
                    "fidelity": target_fidelity,
                    "max_layers": max_layers,
                    "num_restarts": num_restarts,
                    "seed": seed,
                },
                compression_type=CompressionJobType.SU2_EQUIVARIANT_COMPILATION.value,
            )
        )
        return jobs[0]

    @errors.graceful_api_errors_message
    @format_docstring(ARGS=_distribution_loading_args)
    @check_parameters_match(distribution_loading)
    def distribution_loading_estimates(
        self,
        num_qubits: int,
        distribution_name: str,
        interval_start: Real,
        interval_end: Real,
        loc: Real = 0,
        scale: Real = 1,
        encoding: str = "probability",
        num_layers: int = 1,
        truncation_cutoff: Real = 1e-6,
        name: str | None = None,
        job_description: str | None = None,
        **shape,
    ) -> DataLoadingEstimatesModel:
        """Estimate the cost and time of a Data Loading job created by :meth:`distribution_loading`.

        The parameters are the same as for :meth:`distribution_loading`. Once you discover values that result in acceptable cost
        and time estimates, you can remove ``_estimates`` from the end of the method name and call :meth:`distribution_loading`.

        Args:{ARGS}
        Returns:
            DataLoadingEstimatesModel: The estimated time (in seconds) and cost (in Haiqu Credits).

        Examples:
            >>> est = haiqu.distribution_loading_estimates(
            ...     num_qubits=10,
            ...     distribution_name="norm",
            ...     interval_start=-3,
            ...     interval_end=3
            ... )
            >>> est
            DataLoadingEstimatesModel(estimated_time=0.22770169152050562, estimated_cost=0.010079648405964921)
            >>> est.draw()  # in Jupyter notebook
        """
        name, parameters = self._prepare_distribution_loading_params(
            num_qubits,
            distribution_name,
            interval_start,
            interval_end,
            loc,
            scale,
            encoding,
            num_layers,
            truncation_cutoff,
            name,
            shape,
        )
        return self._client.data_loading_estimates(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.DISTRIBUTION_LOADING.value,
                num_qubits=num_qubits,
                parameters=parameters,
            )
        )

    @errors.graceful_api_errors_message
    @format_docstring(ARGS=_vector_loading_args)
    @check_parameters_match(vector_loading)
    def vector_loading_estimates(
        self,
        data: Sequence[Number],
        num_qubits: int | None = None,
        num_layers: int = 2,
        truncation_cutoff: Real = 1e-6,
        fine_tuning_iterations: int = 20,
        max_time: int | float = MAX_DATA_LOADING_TIME,
        name: str | None = None,
        job_description: str | None = None,
    ) -> DataLoadingEstimatesModel:
        """Estimate the cost and time of a Data Loading job created by :meth:`vector_loading`.

        The parameters are the same as for :meth:`vector_loading`. Once you discover values that result in acceptable cost and
        time estimates, you can remove ``_estimates`` from the end of the method name and call :meth:`vector_loading`.

        Args:{ARGS}
        Returns:
            DataLoadingEstimatesModel: The estimated time (in seconds) and cost (in Haiqu Credits).

        Examples:
            >>> est = haiqu.vector_loading_estimates(
            ...     num_qubits=10,
            ...     num_layers=5,
            >>> )
            >>> est
            DataLoadingEstimatesModel(estimated_time=221.07399999999998, estimated_cost=0.06308276)
            >>> est.draw()  # in Jupyter notebook
        """
        name, parameters = self._prepare_vector_loading_params(
            data,
            num_layers,
            truncation_cutoff,
            fine_tuning_iterations,
            max_time,
            name,
        )
        if num_qubits is None:
            num_qubits = int(np.log2(len(parameters["data"])))
        # we pop data vector to save network usage, since qubit number is important, not the data itself
        parameters.pop("data")  # TODO: split estimation and computing models
        return self._client.data_loading_estimates(
            data=DataLoadingSubmitModel(
                dl_type=DataLoadingType.VECTOR_LOADING.value,
                num_qubits=num_qubits,
                parameters=parameters,
            )
        )

    # TODO: enable the method once we have any estimates to return
    # @errors.graceful_api_errors_message
    # def state_compression_estimates(self, num_qubits: int, **parameters) -> StateCompressionEstimatesModel:
    #     """Estimate compression cost and time.
    #
    #     Args:
    #         num_qubits (int): The number of qubits for the wavefunction
    #         num_layers (int): The number of layers
    #         parameters (dict): The optional parameters
    #
    #     Returns:
    #         tuple: The estimated time (Seconds) and the cost (Haiqu Credits).
    #
    #     Example::
    #
    #         # Estimate the cost of compression
    #         haiqu.state_compression_estimates(
    #             num_qubits=10,
    #         ).draw()
    #     """
    #     return self._client.compression_estimates(
    #         data=StateCompressionEstimatesSubmitModel(
    #             num_qubits=num_qubits,
    #             parameters=parameters,
    #         )
    #     )

    @errors.graceful_api_errors_message
    def observable_backpropagation(
        self,
        circuit: Union[QuantumCircuit, CircuitModel],
        observables: Union[SparsePauliOp, list[SparsePauliOp]],
        max_qwc_groups: Optional[int] = 50,
        max_error_total: Optional[float] = 0.05,
        max_error_per_slice: Optional[float] = 0.005,
        log: bool = True,
    ) -> tuple[Union[list[QuantumCircuit], list[CircuitModel]], list[SparsePauliOp]]:
        """Optimize the observables for a circuit with observable backpropagation.

        This method wraps the Qiskit Operator Backpropagation (OBP) functionality to preprocess your circuits and observables for
        efficient execution.

        Args:
            circuit (QuantumCircuit | CircuitModel): The quantum circuit to optimize.
            observables (SparsePauliOp | list(SparsePauliOp)): The observable(s) to optimize. Can be a single ``SparsePauliOp`` or
                                                               list of ``SparsePauliOp``. The order of Pauli terms follows the
                                                               Qiskit reversed-order convention.
            max_qwc_groups (int): Maximum number of qubit-wise commuting groups to create. Defaults to 50.
                Treat with caution as increasing this value may lead to SIGNIFICANTLY higher computational costs!
            max_error_total (float): Maximum error allowed for the entire circuit. Defaults to 0.05.
            max_error_per_slice (float): Maximum error allowed per slice of the circuit. Defaults to 0.005.
            log (bool): If ``True`` (default), logs the reduced circuits to the Haiqu cloud.

        Returns:
            tuple[list, list]: A list of reduced circuits and a list of backpropagated observables.

        Examples:
            >>> from qiskit import QuantumCircuit
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> qc = QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> obs = [SparsePauliOp("ZZ"), SparsePauliOp("XX")]
            >>> optimized_circuits, optimized_obs = haiqu.observable_backpropagation(circuit=qc, observables=obs, log=False)
            >>> [len(qc) for qc in optimized_circuits]
            [0, 0]
            >>> optimized_obs
            [SparsePauliOp(['ZI'],
                           coeffs=[1.+0.j]),
             SparsePauliOp(['IZ'],
                           coeffs=[1.+0.j])]
        """
        if not isinstance(circuit, (QuantumCircuit, CircuitModel)):
            raise ValueError("The 'circuit' argument must be a QuantumCircuit or CircuitModel instance.")

        if isinstance(observables, SparsePauliOp):
            observables = [observables]
        elif not isinstance(observables, SparsePauliOp):
            if isinstance(observables, list):
                if not all(isinstance(obs, SparsePauliOp) for obs in observables):
                    raise ValueError("All elements in 'observables' list must be SparsePauliOp instances.")
            else:
                raise ValueError("The 'observables' argument must be a SparsePauliOp or a list of SparsePauliOp.")

        run_locally = True

        if isinstance(circuit, CircuitModel):
            if not circuit.generated and circuit.qpy is not None:
                circuit = from_qpy(circuit.qpy)
            else:
                run_locally = False

        if isinstance(circuit, QuantumCircuit):
            if is_haiqu_generated(circuit):
                run_locally = False
                circuit = self._get_or_create_circuit(circuit=circuit, name=circuit.name)

        if not run_locally:
            observables = [
                [(str(pauli), float(coeff.real), float(coeff.imag)) for pauli, coeff in zip(obs.paulis, obs.coeffs)]
                for obs in observables
            ]
            submit_data = SubmitObservableBackpropagationModel(
                circuit_id=circuit.id,
                observables=observables,
                max_qwc_groups=max_qwc_groups,
                max_error_total=max_error_total,
                max_error_per_slice=max_error_per_slice,
            )
            response_data = self._client.backpropagate_observables(submit_data=submit_data)
            optimized_circuits = [
                self._client.get_circuit(experiment_id=self._experiment.id, circuit_id=cid)
                for cid in response_data.optimized_circuit_ids
            ]
            backpropagated_observables = response_data.backpropagated_observables

            backpropagated_observables = [
                SparsePauliOp.from_list([(pauli, complex(re_coeff, im_coeff)) for pauli, re_coeff, im_coeff in obs])
                for obs in backpropagated_observables
            ]

            return optimized_circuits, backpropagated_observables

        result_circuits = []
        backpropagated_observables = []

        for i, obs in enumerate(observables):
            bp_circuit, backpropagated_observable = backpropagation(
                circuit,
                obs,
                max_qwc_groups=max_qwc_groups,
                max_error_total=max_error_total,
                max_error_per_slice=max_error_per_slice,
            )

            result_circuits.append(
                self._get_or_create_circuit(circuit=bp_circuit, name=f"obp-{i}-{circuit.name}") if log else bp_circuit
            )
            backpropagated_observables.append(backpropagated_observable)

        return result_circuits, backpropagated_observables

    @errors.graceful_api_errors_message
    def pretrain(
        self,
        problem: Union[VariationalProblem, NonlinearVariationalProblem],
        *,
        max_time: float = 60,
        seed: Optional[int] = 42,
        initial_parameters: Optional[list[float]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> PretrainingJobModel:
        """Pretrain parameters for a variational quantum circuit to minimize the expectation value of input observable.

        Accepts either a linear ``VariationalProblem`` (minimize a single observable's expectation) or a
        ``NonlinearVariationalProblem`` (minimize a sympy objective over several named observables, whose
        terms may include the ``0``/``1`` projector symbols). A linear problem is treated internally as the
        trivial objective ``"x"`` over its single observable.

        For nonlinear problems, Pauli and projector term strings follow Qiskit's reversed-order convention
        (rightmost character = qubit 0). See :class:`~haiqu.sdk.qml.problem.NonlinearVariationalProblem`.

        Args:
            problem (VariationalProblem | NonlinearVariationalProblem): problem instance containing the
                ansatz circuit and either a single observable (linear) or a loss expression with named
                observables (nonlinear).
            max_time (float): maximal time (in seconds) the pretraining can take. If this time exceeds (not counting
                              initialization and other overheads), then the current best result is returned. Defaults to 1 minute.
                              Current maximal pretraining time is 15 minutes.
            seed (int|None): a seed for initial random initialization of weights. They are chosen from uniform
                             distribution in the interval [-π,π). Defaults to 42.
            initial_parameters (list[float]|None): if specified, then these weights are used instead of random ones.
                                                   Defaults to None.
            name (str|None): optional name of the job. If not set, then automatic will be generated.
            description (str|None): optional description of the job. If not set, then automatic will be generated.
        Returns:
            PretrainingJobModel: Job handle to track pretraining progress and retrieve results.
                Call ``job.result()`` to retrieve the pretrained ansatz parameters as a ``list[float]`` (one entry per
                parameter in the input ``VariationalProblem.ansatz``), suitable for use as ``parameters`` in
                :meth:`run` or as ``initial_parameters`` in :meth:`variational_optimization`.
                Run ``help(job.result)`` for the full description of result and ``info`` contents.

        Examples:
            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit.library import efficient_su2
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> from haiqu.sdk.qml import VariationalProblem
            >>> pqc = QuantumCircuit(5)
            >>> pqc.compose(efficient_su2(num_qubits=pqc.num_qubits, reps=1), inplace=True)
            >>> loss = SparsePauliOp(["ZIIII", "IIZXI"])
            >>> problem = VariationalProblem(pqc, loss)
            >>> job = haiqu.pretrain(problem, max_time=10)
            >>> pretrained_params = job.result()  # accessing the pretrained parameters
            >>> haiqu.run([problem.ansatz], observables=[problem.observable], parameters=[pretrained_params],
            ...           device=haiqu.get_device("aer_simulator")).result()  # checking the result
            [[[-2.0]]]  # result may vary. -2 is the optimal loss for two independent pauli strings

            Nonlinear objective over several observables (terms may include the ``0``/``1`` projectors):

            >>> from haiqu.sdk.qml import NonlinearVariationalProblem
            >>> problem = NonlinearVariationalProblem(
            ...     pqc, "1 - x/y", {"x": [("ZIIII", 1.0)], "y": [("0IIII", 0.5), ("1IIZI", -0.5)]}
            ... )
            >>> job = haiqu.pretrain(problem, max_time=10)
        """
        self._check_experiment()

        ansatz, loss_expression, observables = _prepare_nonlinear_problem(problem)

        # Log the ansatz circuit
        circuit = self._get_or_create_circuit(ansatz)

        if name is None:
            name = f"pretrain-{circuit.id}-{circuit.name}"
        if description is None:
            description = (
                f"Pretraining of {circuit.name} with a loss over {len(observables)} observable(s) " f"for {max_time} seconds."
            )
        if initial_parameters is not None:
            initial_parameters = list(initial_parameters)

        return self._client.pretraining(
            data=PretrainingSubmitModel(
                experiment_id=self._experiment.id,
                circuit_id=circuit.id,
                loss_expression=loss_expression,
                observables=observables,
                max_time=max_time,
                seed=seed,
                initial_parameters=initial_parameters,
                name=name,
                description=description,
                pretrain_type=PretrainingJobType.PRETRAIN.value,
            )
        )

    @errors.graceful_api_errors_message
    def gradient(
        self,
        problem: Union[VariationalProblem, NonlinearVariationalProblem],
        weights: list[float],
        *,
        max_bond_dimension: int = 40,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> PretrainingJobModel:
        """Compute the loss and gradient vector of a parametrized circuit at given weights.
        Result is returned as a tuple (loss, gradient) and compatible with various optimizers,
        which expect the gradient input such as scipy's minimize with ``jac=True``.

        Accepts either a linear ``VariationalProblem`` or a ``NonlinearVariationalProblem`` (a sympy
        objective over several named observables whose terms may include the ``0``/``1`` projector
        symbols). A linear problem is treated internally as the trivial objective ``"x"`` over its
        single observable.

        For nonlinear problems, Pauli and projector term strings follow Qiskit's reversed-order convention
        (rightmost character = qubit 0). See :class:`~haiqu.sdk.qml.problem.NonlinearVariationalProblem`.

        Args:
            problem (VariationalProblem | NonlinearVariationalProblem): problem instance containing the
                ansatz circuit and either a single observable (linear) or a loss expression with named
                observables (nonlinear).
            weights (list[float]): parameter values at which to evaluate the loss and gradient.
                Must match the parameters in the ansatz circuit, in the order returned by
                Qiskit's ``QuantumCircuit.parameters``.
            max_bond_dimension (int): maximum bond dimension for the MPS representation. Defaults to 40.
                Maximum allowed value is 256.
            name (str|None): optional name of the job. If not set, then automatic will be generated.
            description (str|None): optional description of the job. If not set, then automatic will be generated.

        Returns:
            PretrainingJobModel: Job handle to track progress and retrieve results.
                Call ``job.result()`` to retrieve a ``(loss, gradient)`` tuple, where ``loss``
                is the observable expectation value (float) and ``gradient`` is a list of partial
                derivatives (one float per parameter in the ansatz).

        Examples:
            >>> import numpy as np
            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit.library import efficient_su2
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> from haiqu.sdk.qml import VariationalProblem
            >>> pqc = QuantumCircuit(2)
            >>> pqc.compose(efficient_su2(num_qubits=2, reps=1), inplace=True)
            >>> problem = VariationalProblem(pqc, SparsePauliOp(["ZI", "IZ"]))
            >>> weights = [0.1] * pqc.num_parameters
            >>> job = haiqu.gradient(problem, weights)
            >>> loss, grad = job.result()
            >>> print(loss)
            1.9642185222880129
            >>> print(np.round(grad, 3))
            [-0.208 -0.207  0.     0.001 -0.109 -0.198  0.     0.   ]
        """
        self._check_experiment()

        ansatz, loss_expression, observables = _prepare_nonlinear_problem(problem)

        # Log the ansatz circuit
        circuit = self._get_or_create_circuit(ansatz)

        if name is None:
            name = f"gradient-{circuit.id}-{circuit.name}"
        if description is None:
            description = f"Gradient of {circuit.name} with a loss over {len(observables)} observable(s)."

        return self._client.pretraining(
            data=PretrainingSubmitModel(
                experiment_id=self._experiment.id,
                circuit_id=circuit.id,
                loss_expression=loss_expression,
                observables=observables,
                initial_parameters=list(weights),
                options={"max_bond_dimension": max_bond_dimension},
                name=name,
                description=description,
                pretrain_type=PretrainingJobType.MPS_GRADIENT.value,
            )
        )

    @errors.graceful_api_errors_message
    def variational_optimization(
        self,
        problem: VariationalProblem,
        shots: int = 1000,
        device: DeviceModel | None = None,
        device_id: str | None = None,
        options: dict | None = None,
        initial_parameters: Optional[list[float]] = None,
        seed: Optional[int] = None,
        optimizer_options: Optional[OptimizerOptions] = None,
        use_mitigation: bool = False,
        use_packing: bool = False,
        pack_size: Optional[int] = None,
        use_session: bool = False,
        use_compression: bool = False,
        compression_options: Optional[CompressionOptions] = None,
        job_name: str | None = None,
        job_description: str | None = None,
        dry_run: bool = False,
    ) -> VariationalJobModel:
        """Optimize a variational quantum circuit to minimize the expectation value of input observable.

        Defaults to the NFT (Nakanishi-Fujii-Todo) optimizer, a gradient-free optimizer
        designed for variational quantum algorithms (https://arxiv.org/abs/1903.12166).
        Pass a ``ScipyOptimizerOptions`` instance as ``optimizer_options`` to dispatch
        to any derivative-free ``scipy.optimize.minimize`` method instead (``cobyla``,
        ``nelder-mead``, ``powell``, ``cobyqa``).

        Args:
            problem (VariationalProblem): problem instance containing the ansatz circuit and observable.
            shots: Number of shots per circuit evaluation. Defaults to 1000.
            device: Device to execute on. If specified, device_id is ignored. Devices from
                :meth:`get_device` with ``use_fractional_gates=True`` use fractional gates.
            device_id: ID of the device to execute on. Defaults to None.
            options: Additional device options. Hybrid programs can pass
                ``DeviceLayer(options={"use_fractional_gates": True})``.
            initial_parameters: Initial parameter values. Cannot be used together with seed.
                If neither is provided, random parameters in [-0.1π, 0.1π] are generated.
            seed: Random seed for reproducible generation of initial parameters from a uniform
                distribution in [-0.1π, 0.1π]. Cannot be used together with initial_parameters.
            optimizer_options: Configuration for the optimizer. If None, defaults to
                NFTOptimizerOptions(). Pass a ScipyOptimizerOptions instance to use any
                derivative-free scipy method (cobyla, nelder-mead, powell, cobyqa) instead.
            use_mitigation: Whether to use error mitigation techniques. Defaults to False.
            use_packing: Whether to use circuit packing for efficient device utilization. Defaults to False.
                **Warning:** Experimental — packing replicates circuits on unused device qubits
                to run multiple copies in parallel, which may increase errors for deeper input circuits.
                For example, a 4-qubit circuit with pack_size=2 and 1000 shots runs two copies
                in parallel with 500 shots each, yielding 1000 shots of results while only paying
                for 500 shot executions on the QPU — a 2x cost saving.
            pack_size: Number of circuit copies to pack onto the device. Must be >= 2.
                Only valid when ``use_packing=True``. If ``None`` (default), the backend will
                pack into at most 2/3 of the device qubits.
            use_session: Whether to use IBM Qiskit Runtime Session for execution. Defaults to False.
            use_compression: Whether to apply circuit compression at each training step.
                Binds parameters and compresses the circuit before each QPU evaluation, reducing
                2-qubit gate count and thus QPU noise. Defaults to False.
            compression_options: Configuration for compression-in-training. Only used when
                ``use_compression=True``. If ``None``, default compression settings are applied.
            job_name: The name for the job. If ``None`` (default), a name will be automatically generated.
            job_description: The description for the job.
            dry_run (bool): Whether to stop just prior to backend execution for QPU cost estimation. Defaults to ``False``.
                When ``True``, the job's optimization result will be empty since execution on the device is skipped.
                The estimated QPU cost is then available via ``job.estimated_qpu_cost``. When ``use_session=True``,
                the estimate excludes classical optimization and parameter-update time, which session mode also bills;
                ``job.estimated_qpu_cost["warning"]`` carries this notice. Only supported for the NFT optimizer;
                passing a ``ScipyOptimizerOptions`` with ``dry_run=True`` raises ``NotImplementedError``.

        Returns:
            VariationalJobModel: Job handle to track optimization progress and retrieve results.
                Call ``job.result()`` to retrieve a ``VariationalResult`` exposing ``optimal_parameters`` (``list[float]``),
                ``min_loss`` (``float``), ``loss_history`` (``list[float]``), and ``weights_history`` (``list[list[float]]``).
                ``job.info`` exposes auxiliary metadata (``loss_history``, ``qpu_cost``, ``session_cost``).
                When ``dry_run=True``, ``result()`` is empty; use ``job.estimated_qpu_cost`` instead.
                Use ``job.progress()`` for live status updates and ``help(job.result)`` for the full description of result
                and ``info`` contents.

        Examples:
            Default optimizer settings:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> from haiqu.sdk.qml import VariationalProblem
            >>> theta = Parameter('θ')
            >>> ansatz = QuantumCircuit(2)
            >>> ansatz.ry(theta, 0)
            >>> ansatz.cx(0, 1)
            >>> obs = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5)])
            >>> problem = VariationalProblem(ansatz, obs)
            >>> job = haiqu.variational_optimization(problem, shots=1000, device_id="aer_simulator")
            >>> result = job.result()
            >>> print(result.min_loss)

            Custom optimizer settings:

            >>> from haiqu.sdk.qml import NFTOptimizerOptions
            >>> optimizer = NFTOptimizerOptions(maxfev=2048, maxiter=100)
            >>> job = haiqu.variational_optimization(problem, shots=1000, device_id="aer_simulator", optimizer_options=optimizer)

            Scipy COBYLA instead of NFT:

            >>> from haiqu.sdk.qml import ScipyOptimizerOptions
            >>> optimizer = ScipyOptimizerOptions(
            ...     method="cobyla",
            ...     maxfev=2000,
            ...     options={"rhobeg": 0.5, "tol": 1e-8, "maxiter": 2000},
            ... )
            >>> job = haiqu.variational_optimization(problem, shots=1000, device_id="aer_simulator", optimizer_options=optimizer)
        """
        self._check_experiment()

        if not isinstance(problem, VariationalProblem):
            raise TypeError("problem must be a VariationalProblem instance.")

        # Validate that only one of seed or initial_parameters is provided
        if seed is not None and initial_parameters is not None:
            raise ValueError("Cannot specify both 'seed' and 'initial_parameters'. Use only one.")

        # Generate initial parameters from seed if provided, or random if neither is specified
        if seed is not None:
            rng = np.random.default_rng(seed)
            initial_parameters = rng.uniform(-0.1 * np.pi, 0.1 * np.pi, problem.num_parameters).tolist()
        elif initial_parameters is None:
            initial_parameters = np.random.uniform(-0.1 * np.pi, 0.1 * np.pi, problem.num_parameters).tolist()

        # Handle device selection (same pattern as run())
        if device is not None:
            device_id = device.id
        if device_id is None:
            raise ValueError("Either device or device_id must be specified.")

        options = {} if options is None else copy.deepcopy(options)

        if optimizer_options is None:
            optimizer_options = NFTOptimizerOptions()
        elif not isinstance(optimizer_options, OptimizerOptions):
            raise TypeError("optimizer_options must be an OptimizerOptions subclass (e.g., NFTOptimizerOptions).")

        if dry_run and not isinstance(optimizer_options, NFTOptimizerOptions):
            raise NotImplementedError(
                "dry_run QPU cost estimation is only supported for the NFT optimizer (NFTOptimizerOptions)."
            )

        # Warn if parameters are shared across multiple gates (violates NFT precondition 1)
        if isinstance(optimizer_options, NFTOptimizerOptions):
            shared_params = find_shared_parameters(problem.ansatz)
            if shared_params:
                warnings.warn(
                    f"Parameters {shared_params} appear to be used in multiple gates. "
                    "The NFT optimizer requires each parameter to appear in exactly one gate, "
                    "so results may be incorrect. "
                    "If you did not intentionally share parameters, this warning may be due to "
                    "transpilation which can decompose single parameterized gates into multiple gates "
                    "that reuse the same parameter. In that case, consider passing the untranspiled "
                    "ansatz circuit instead.",
                    UserWarning,
                    stacklevel=2,
                )

        # Log the ansatz circuit
        circuit = self._get_or_create_circuit(problem.ansatz)

        if job_name is None:
            job_name = "variational-optim-job-" + circuit.name

        # Extract observable as tuple of (pauli_strings, coefficients) - same format as run()
        pauli_strings = [str(pauli) for pauli in problem.observable.paulis]
        coefficients = [float(coeff.real) for coeff in problem.observable.coeffs]
        observable = (pauli_strings, coefficients)

        if use_session:
            options["use_session"] = True

        if device is not None and device.use_fractional_gates:
            options["use_fractional_gates"] = True

        if dry_run:
            options["dry_run"] = True

        # Validate and pass packing options
        self._validate_packing(use_packing, pack_size)
        options["use_packing"] = use_packing
        if pack_size is not None:
            options["pack_size"] = pack_size

        if use_compression:
            options["use_compression"] = True
            options["compression_options"] = compression_options.model_dump() if compression_options is not None else {}

        submit_data = VariationalProblemSubmitModel(
            experiment_id=self._experiment.id,
            circuit_id=circuit.id,
            observable=observable,
            shots=shots,
            device_id=device_id,
            options=options,
            initial_parameters=initial_parameters,
            optimizer_options=optimizer_options,
            use_mitigation=use_mitigation,
            name=job_name,
            description=job_description,
        )

        return self._client.variational_optimization(data=submit_data)

    @errors.graceful_api_errors_message
    def flow(
        self,
        program: HybridProgram,
        circuits: QuantumCircuit | list[QuantumCircuit] | CircuitModel | list[CircuitModel],
        shots: int = 1000,
        parameters: list | None = None,
        observables: SparsePauliOp | list[SparsePauliOp] | list[list[SparsePauliOp]] | None = None,
        job_name: str | None = None,
        job_description: str | None = None,
        device_credentials: dict | None = None,
        dry_run: bool = False,
    ) -> HybridJobModel:
        """
        Run a flow (hybrid program).

        This flexible method supports multiple execution scenarios, with different combinations of circuits, parameters, and
        observables. When multiple values are provided for any of them, the results are returned as nested lists with up to 3
        layers, ordered by circuits, then observables, and finally parameters.

        Args:
            program (HybridProgram): The hybrid program to execute.
            circuits: The quantum circuit(s) to pass to the hybrid program. Can be a single circuit or a list of circuits.
            shots (int): The number of shots for each circuit execution. Defaults to 1000.
            parameters: The parameters for the circuits. Can be a single set of parameters or nested lists of parameter sets. For
                        multiple circuits, must be a list where each element corresponds to parameters for that circuit. Defaults
                        to ``None``, in which case the circuits must not have any parameters.

                        Each inner list must have length ``circuit.num_parameters``. Values are bound positionally in the
                        order of Qiskit's ``QuantumCircuit.parameters``. When parameters are created one-by-one, that order
                        is typically alphabetical by name (e.g. ``theta10`` before ``theta2``),
                        independent of gate addition order; ``ParameterVector`` circuits use vector
                        index order instead (e.g. ``theta[0]``, ``theta[1]``). Inspect
                        ``list(circuit.parameters)`` for the exact order.
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
            job_name (str | None): The name for the job. If ``None`` (default), a name will be automatically generated.
            job_description (str | None): The description for the job.
            device_credentials (dict | None): Credentials for device access.
            dry_run (bool): Whether to stop just prior to backend execution for QPU cost estimation. Defaults to ``False``.
                When ``True``, the job result will be empty since execution on the device is skipped.
                The estimated QPU cost is then available via ``job.estimated_qpu_cost``.
                Wall-clock time to run hybrid program layers up to (but not including) the device
                layer is available via ``job.pre_device_pipeline_time`` (also on full runs).

        Returns:
            HybridJobModel: The Hybrid job that will execute the hybrid program.
                Call ``job.result()`` to retrieve the execution results as a nested list ordered by
                *circuits → observables → parameters*:

                * Without observables: list of measurement distributions (``dict[str, float]``), one per
                  circuit, in Qiskit bit-order.
                * With observables, no parameter sweep: 2D list of expectation values, indexed ``[circuit][observable]``.
                * With observables and a parameter sweep: 3D list of expectation values, indexed
                  ``[circuit][observable][parameter]``.

                **Dry runs** (``dry_run=True``): ``result()`` is empty. Use ``job.estimated_qpu_cost`` for
                the QPU cost estimate and ``job.pre_device_pipeline_time`` for classical hybrid time
                before the device layer. ``job.time`` is ``0`` because the device phase does not run.

                **Full runs** (``dry_run=False``): ``job.time`` is the device-phase wall clock (from when
                device execution starts to job completion). ``job.pre_device_pipeline_time`` reports
                classical hybrid time before the device layer. ``job.info`` also exposes auxiliary
                metadata (``uncertainty`` when observables are supplied, ``qpu_cost`` on full runs).
                Run ``help(job.result)`` for the full description of result and ``info`` contents.

        Examples:
            Single circuit, no parameters, no observables:

            >>> from qiskit import QuantumCircuit
            >>> from haiqu.sdk.hybrid import HybridProgram, layers
            >>> program = HybridProgram(layers=[
            ...     layers.InputLayer(),
            ...     layers.DeviceLayer(device_id="aer_simulator"),
            ... ])
            >>> qc = QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> qc.measure_all()
            >>> job = haiqu.flow(program, circuits=qc)
            >>> job.result()  # Returns: [dist_c1] (bitstrings in Qiskit convention)
            [{'00': 0.504, '11': 0.496}]

            Single circuit, multiple parameters, no observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> from haiqu.sdk.hybrid import HybridProgram, layers
            >>> program = HybridProgram(layers=[
            ...     layers.InputLayer(),
            ...     layers.DeviceLayer(device_id="aer_simulator"),
            ... ])
            >>> theta = Parameter('θ')
            >>> qc = QuantumCircuit(2)
            >>> qc.ry(theta, 0)
            >>> qc.cx(0, 1)
            >>> qc.measure_all()
            >>> job = haiqu.flow(
            ...     program,
            ...     circuits=qc,
            ...     parameters=[[0.5], [1.0]],
            ... )
            >>> job.result()  # Returns: [[dist_c1_p1, dist_c1_p2]]
            [[{'00': 0.934, '11': 0.066}, {'00': 0.802, '11': 0.198}]]

            Single circuit, no parameters, multiple observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> from haiqu.sdk.hybrid import HybridProgram, layers
            >>> program = HybridProgram(layers=[
            ...     layers.InputLayer(),
            ...     layers.EstimatorLayer(),
            ...     layers.DeviceLayer(device_id="aer_simulator"),
            ... ])
            >>> qc = QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> obs = [SparsePauliOp("ZZ"), SparsePauliOp("XY")]
            >>> job = haiqu.flow(
            ...     program,
            ...     circuits=qc,
            ...     observables=obs,
            ... )
            >>> job.result()  # Returns: [[exp_c1_obs1, exp_c1_obs2]]
            [[1.0, 0.018000000000000016]]

            Single circuit, multiple parameters, multiple observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> from haiqu.sdk.hybrid import HybridProgram, layers
            >>> program = HybridProgram(layers=[
            ...     layers.InputLayer(),
            ...     layers.EstimatorLayer(),
            ...     layers.DeviceLayer(device_id="aer_simulator"),
            ... ])
            >>> theta = Parameter('θ')
            >>> qc = QuantumCircuit(2)
            >>> qc.ry(theta, 0)
            >>> qc.cx(0, 1)
            >>> params = [[0.5], [1.0]]
            >>> obs = [SparsePauliOp("ZZ"), SparsePauliOp("XX")]
            >>> job = haiqu.flow(
            ...     program,
            ...     circuits=qc,
            ...     parameters=params,
            ...     observables=obs,
            ... )
            >>> job.result()  # Returns: [[[exp_c1_obs1_p1, exp_c1_obs1_p2], [exp_c1_obs2_p1, exp_c1_obs2_p2]]]
            [[[1.0, 1.0], [0.49, 0.846]]]

            Multiple circuits, no parameters, no observables:

            >>> from qiskit import QuantumCircuit
            >>> from haiqu.sdk.hybrid import HybridProgram, layers
            >>> program = HybridProgram(layers=[
            ...     layers.InputLayer(),
            ...     layers.DeviceLayer(device_id="aer_simulator"),
            ... ])
            >>> qc1 = QuantumCircuit(2)
            >>> qc1.h(0)
            >>> qc1.cx(0, 1)
            >>> qc1.measure_all()
            >>> qc2 = QuantumCircuit(2)
            >>> qc2.x(0)
            >>> qc2.cx(0, 1)
            >>> qc2.measure_all()
            >>> circuits = [qc1, qc2]
            >>> job = haiqu.flow(program, circuits=circuits)
            >>> job.result()  # Returns: [dist_c1, dist_c2]
            [{'11': 0.524, '00': 0.476}, {'11': 1.0}]

            Multiple circuits, multiple parameters, no observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> from haiqu.sdk.hybrid import HybridProgram, layers
            >>> program = HybridProgram(layers=[
            ...     layers.InputLayer(),
            ...     layers.DeviceLayer(device_id="aer_simulator"),
            ... ])
            >>> theta = Parameter('θ')
            >>> qc1 = QuantumCircuit(2)
            >>> qc1.ry(theta, 0)
            >>> qc1.cx(0, 1)
            >>> qc1.measure_all()
            >>> qc2 = QuantumCircuit(2)
            >>> qc2.rx(theta, 0)
            >>> qc2.cz(0, 1)
            >>> qc2.measure_all()
            >>> circuits = [qc1, qc2]
            >>> params = [[[0.5], [1.0]], [[0.3], [0.7]]]  # Parameters for each circuit
            >>> job = haiqu.flow(
            ...     program,
            ...     circuits=circuits,
            ...     parameters=params,
            ... )
            >>> job.result()  # Returns: [[dist_c1_p1, dist_c1_p2], [dist_c2_p1, dist_c2_p2]]
            [[{'00': 0.955, '11': 0.045}, {'00': 0.783, '11': 0.217}],
             [{'00': 0.982, '01': 0.018}, {'00': 0.882, '01': 0.118}]]

            Multiple circuits, no parameters, multiple observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> from haiqu.sdk.hybrid import HybridProgram, layers
            >>> program = HybridProgram(layers=[
            ...     layers.InputLayer(),
            ...     layers.EstimatorLayer(),
            ...     layers.DeviceLayer(device_id="aer_simulator"),
            ... ])
            >>> qc1 = QuantumCircuit(2)
            >>> qc1.h(0)
            >>> qc1.cx(0, 1)
            >>> qc2 = QuantumCircuit(2)
            >>> qc2.x(0)
            >>> qc2.cx(0, 1)
            >>> circuits = [qc1, qc2]
            >>> obs = [[SparsePauliOp("ZZ"), SparsePauliOp("XX")],
            ...        [SparsePauliOp("YY"), SparsePauliOp("ZX")]]  # Observables for each circuit
            >>> job = haiqu.flow(
            ...     program,
            ...     circuits=circuits,
            ...     observables=obs,
            ... )
            >>> job.result()  # Returns: [[exp_c1_obs1, exp_c1_obs2], [exp_c2_obs1, exp_c2_obs2]]
            [[1.0, 1.0], [0.0, -0.0020000000000000018]]

            Multiple circuits, multiple parameters, multiple observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> from haiqu.sdk.hybrid import HybridProgram, layers
            >>> program = HybridProgram(layers=[
            ...     layers.InputLayer(),
            ...     layers.EstimatorLayer(),
            ...     layers.DeviceLayer(device_id="aer_simulator"),
            ... ])
            >>> theta = Parameter('θ')
            >>> qc1 = QuantumCircuit(2)
            >>> qc1.ry(theta, 0)
            >>> qc1.cx(0, 1)
            >>> qc2 = QuantumCircuit(2)
            >>> qc2.rx(theta, 0)
            >>> qc2.cz(0, 1)
            >>> circuits = [qc1, qc2]
            >>> params = [[[0.5], [1.0]], [[0.3], [0.7]]]  # Parameters for each circuit
            >>> obs = [[SparsePauliOp("ZZ"), SparsePauliOp("XX")],
            ...        [SparsePauliOp("YY"), SparsePauliOp("ZX")]]  # Observables for each circuit
            >>> job = haiqu.flow(
            ...     program,
            ...     circuits=circuits,
            ...     parameters=params,
            ...     observables=obs,
            ... )
            >>> job.result()
            # Returns: [
            #     [[exp_c1_obs1_p1, exp_c1_obs1_p2], [exp_c1_obs2_p1, exp_c1_obs2_p2]],
            #     [[exp_c2_obs1_p1, exp_c2_obs1_p2], [exp_c2_obs2_p1, exp_c2_obs2_p2]],
            # ]
            [[[1.0, 1.0], [0.482, 0.8280000000000001]],
             [[-0.016000000000000014, 0.003999999999999963],
              [-0.02400000000000002, 0.008000000000000007]]]
        """
        self._check_experiment()
        logged_circuits = self._prepare_circuits(circuits)

        if job_name is None:
            job_name = job_name_from_circuits(logged_circuits)

        device_id = None
        new_layers = []

        for layer in program.layers:
            match layer:
                case layers.DeviceLayer():
                    if device_id is not None:
                        raise ValueError("More than one DeviceLayer found in HybridProgram")

                    device_id = layer.device_id
                    options = copy.deepcopy(layer.options)

                    self._normalize_noise_model_option(device_id=device_id, options=options)

                    new_layer = layers.DeviceLayer.model_validate(layer.model_copy(update={"options": options}))
                    new_layers.append(new_layer)
                case _:
                    new_layers.append(layer)

        program = HybridProgram.model_validate(program.model_copy(update={"layers": new_layers}))

        if device_id is None:
            raise ValueError("No DeviceLayer in HybridProgram")

        # Validate that device exists
        self.get_device(device_id=device_id)

        device_credentials = copy.deepcopy(device_credentials) if device_credentials is not None else {}

        if not dry_run:
            self._inject_device_credentials(device_id, device_credentials)

        parameters, observables = validate_and_normalize_parameters_and_observables(
            parameters,
            observables,
            len(logged_circuits),
            num_parameters_per_circuit=get_num_parameters_per_circuit(circuits),
        )

        job = self._client.flow(
            data=HybridSubmitModel(
                experiment_id=self._experiment.id,
                name=job_name,
                description=job_description,
                program=program,
                circuit_ids=[c.id for c in logged_circuits],
                shots=shots,
                parameters=parameters,
                observables=observables,
                device_credentials=device_credentials,
                dry_run=dry_run,
            )
        )

        return job

    # TODO: reimplement haiqu.run using the flow endpoint
    @errors.graceful_api_errors_message
    def run(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit] | CircuitModel | list[CircuitModel],
        parameters: list | None = None,
        shots: int = 1000,
        observables: SparsePauliOp | list[SparsePauliOp] | list[list[SparsePauliOp]] | None = None,
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
        """Run quantum circuits on the selected backend.

        This flexible method supports multiple execution scenarios, with different combinations of circuits, parameters, and
        observables. When multiple values are provided for any of them, the results are returned as nested lists with up to 3
        layers, ordered by circuits, then observables, and finally parameters.

        .. note::

            Every ``device_id`` — including ``"aer_simulator"`` — dispatches to the Haiqu cloud; there is no
            local execution path, and :meth:`login` must have succeeded first. Jobs submitted before
            :meth:`init` is called are logged under a shared default experiment; call
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
            device (DeviceModel | None): The device to run the circuits on. If specified, ``device_id`` is ignored.
                Devices from :meth:`get_device` with ``use_fractional_gates=True`` use fractional gates.
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

                See `the run reference <https://docs.haiqu.ai/reference/run/run>`_ for full details.

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

        Examples:
            Single circuit, no parameters, no observables:

            >>> from qiskit import QuantumCircuit
            >>> qc = QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> qc.measure_all()
            >>> job = haiqu.run(circuits=qc, device_id="aer_simulator")
            >>> job.result()  # Returns: [dist_c1] (bitstrings in Qiskit convention)
            [{'00': 0.504, '11': 0.496}]

            Single circuit, multiple parameters, no observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> theta = Parameter('θ')
            >>> qc = QuantumCircuit(2)
            >>> qc.ry(theta, 0)
            >>> qc.cx(0, 1)
            >>> qc.measure_all()
            >>> job = haiqu.run(
            ...     circuits=qc,
            ...     parameters=[[0.5], [1.0]],
            ...     device_id="aer_simulator",
            ... )
            >>> job.result()  # Returns: [[dist_c1_p1, dist_c1_p2]]
            [[{'00': 0.934, '11': 0.066}, {'00': 0.802, '11': 0.198}]]

            Single circuit, no parameters, multiple observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> qc = QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> obs = [SparsePauliOp("ZZ"), SparsePauliOp("XY")]
            >>> job = haiqu.run(
            ...     circuits=qc,
            ...     observables=obs,
            ...     device_id="aer_simulator",
            ... )
            >>> job.result()  # Returns: [[exp_c1_obs1, exp_c1_obs2]]
            [[1.0, 0.018000000000000016]]

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
            >>> job = haiqu.run(
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
            >>> circuits = [qc1, qc2]
            >>> job = haiqu.run(circuits=circuits, device_id="aer_simulator")
            >>> job.result()  # Returns: [dist_c1, dist_c2]
            [{'11': 0.524, '00': 0.476}, {'11': 1.0}]

            Multiple circuits, multiple parameters, no observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> theta = Parameter('θ')
            >>> qc1 = QuantumCircuit(2)
            >>> qc1.ry(theta, 0)
            >>> qc1.cx(0, 1)
            >>> qc1.measure_all()
            >>> qc2 = QuantumCircuit(2)
            >>> qc2.rx(theta, 0)
            >>> qc2.cz(0, 1)
            >>> qc2.measure_all()
            >>> circuits = [qc1, qc2]
            >>> params = [[[0.5], [1.0]], [[0.3], [0.7]]]  # Parameters for each circuit
            >>> job = haiqu.run(
            ...     circuits=circuits,
            ...     parameters=params,
            ...     device_id="aer_simulator",
            ... )
            >>> job.result()  # Returns: [[dist_c1_p1, dist_c1_p2], [dist_c2_p1, dist_c2_p2]]
            [[{'00': 0.955, '11': 0.045}, {'00': 0.783, '11': 0.217}],
             [{'00': 0.982, '01': 0.018}, {'00': 0.882, '01': 0.118}]]

            Multiple circuits, no parameters, multiple observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> qc1 = QuantumCircuit(2)
            >>> qc1.h(0)
            >>> qc1.cx(0, 1)
            >>> qc2 = QuantumCircuit(2)
            >>> qc2.x(0)
            >>> qc2.cx(0, 1)
            >>> circuits = [qc1, qc2]
            >>> obs = [[SparsePauliOp("ZZ"), SparsePauliOp("XX")],
            ...        [SparsePauliOp("YY"), SparsePauliOp("ZX")]]  # Observables for each circuit
            >>> job = haiqu.run(
            ...     circuits=circuits,
            ...     observables=obs,
            ...     device_id="aer_simulator",
            ... )
            >>> job.result()  # Returns: [[exp_c1_obs1, exp_c1_obs2], [exp_c2_obs1, exp_c2_obs2]]
            [[1.0, 1.0], [0.0, -0.0020000000000000018]]

            Multiple circuits, multiple parameters, multiple observables:

            >>> from qiskit import QuantumCircuit
            >>> from qiskit.circuit import Parameter
            >>> from qiskit.quantum_info import SparsePauliOp
            >>> theta = Parameter('θ')
            >>> qc1 = QuantumCircuit(2)
            >>> qc1.ry(theta, 0)
            >>> qc1.cx(0, 1)
            >>> qc2 = QuantumCircuit(2)
            >>> qc2.rx(theta, 0)
            >>> qc2.cz(0, 1)
            >>> circuits = [qc1, qc2]
            >>> params = [[[0.5], [1.0]], [[0.3], [0.7]]]  # Parameters for each circuit
            >>> obs = [[SparsePauliOp("ZZ"), SparsePauliOp("XX")],
            ...        [SparsePauliOp("YY"), SparsePauliOp("ZX")]]  # Observables for each circuit
            >>> job = haiqu.run(
            ...     circuits=circuits,
            ...     parameters=params,
            ...     observables=obs,
            ...     device_id="aer_simulator",
            ... )
            >>> job.result()
            # Returns: [
            #     [[exp_c1_obs1_p1, exp_c1_obs1_p2], [exp_c1_obs2_p1, exp_c1_obs2_p2]],
            #     [[exp_c2_obs1_p1, exp_c2_obs1_p2], [exp_c2_obs2_p1, exp_c2_obs2_p2]],
            # ]
            [[[1.0, 1.0], [0.482, 0.8280000000000001]],
             [[-0.016000000000000014, 0.003999999999999963],
              [-0.02400000000000002, 0.008000000000000007]]]

            Example of using Matrix Product State (MPS) simulator for larger scale experiments:

            >>> from qiskit.circuit.random import random_circuit
            >>> circuit = random_circuit(num_qubits=20, depth=3, max_operands=3, seed=2025, measure=True)
            >>> job = haiqu.run(circuit, device_id="aer_simulator", options={
            ...                 "method": "matrix_product_state",  # set method to MPS
            ...                 "matrix_product_state_max_bond_dimension": 16,  # preferably, limit the bonds
            ...               })
        """
        self._check_experiment()

        options = {} if options is None else copy.deepcopy(options)

        # Validate error_mitigation_options if provided
        if "error_mitigation_options" in options:
            emo = options["error_mitigation_options"]
            if not isinstance(emo, dict):
                raise ValueError("`error_mitigation_options` in `options` must be a dictionary.")
            dict_type_keys = {"readout_mitigation_options"}
            bool_type_keys = {
                "dynamical_decoupling",
                "readout_mitigation",
                "noise_tailoring",
                "advanced_mitigation",
            }
            valid_emo_keys = dict_type_keys | bool_type_keys
            unknown_keys = set(emo.keys()) - valid_emo_keys
            if unknown_keys:
                raise ValueError(
                    f"Unknown key(s) in `error_mitigation_options`: {unknown_keys}. " f"Valid keys are: {valid_emo_keys}."
                )
            for key, val in emo.items():
                if key in dict_type_keys:
                    if not isinstance(val, dict):
                        raise ValueError(f"{key} must be a `dict`." f"Got {type(val).__name__!r} for key '{key}'.")
                elif key in bool_type_keys:
                    if not isinstance(val, bool):
                        raise ValueError(f"{key} must be a `bool`." f"Got {type(val).__name__!r} for key '{key}'.")
            if not use_mitigation:
                warnings.warn(
                    "`error_mitigation_options` provided but `use_mitigation=False`. "
                    "Mitigation options will have no effect unless `use_mitigation=True`.",
                    UserWarning,
                    stacklevel=2,
                )
            if not emo.get("readout_mitigation", False) and emo.get("readout_mitigation_options", None):
                warnings.warn(
                    "`readout_mitigation_options` provided but `readout_mitigation=False`. "
                    "Readout mitigation options will have no effect unless `readout_mitigation=True`.",
                    UserWarning,
                    stacklevel=2,
                )

        # Validate device ID
        if device is None:
            if device_id is None:
                raise ValueError("The `device` or `device_id` is required to run the circuits.")
            else:
                # Validate that device exists
                self.get_device(device_id=device_id)

        # If both device and device_id are provided, device_id is ignored
        if device is not None:
            device_id = device.id

        self._normalize_noise_model_option(device_id=device_id, options=options)

        if not dry_run:
            self._inject_device_credentials(device_id, options)

        logged_circuits = self._prepare_circuits(circuits)

        if job_name is None:
            job_name = job_name_from_circuits(logged_circuits)

        parameters, observables = validate_and_normalize_parameters_and_observables(
            parameters,
            observables,
            len(logged_circuits),
            num_parameters_per_circuit=get_num_parameters_per_circuit(circuits),
        )

        # Validate and pass packing options
        self._validate_packing(use_packing, pack_size)
        options["use_packing"] = use_packing
        if pack_size is not None:
            options["pack_size"] = pack_size

        if device is not None and device.use_fractional_gates:
            options["use_fractional_gates"] = True

        # Submit the job
        job = self._client.run(
            data=RunSubmitModel(
                experiment_id=self._experiment.id,
                circuit_ids=[c.id for c in logged_circuits],
                parameters=parameters,
                shots=shots,
                observables=observables,
                device_id=device_id,
                options=options,
                use_mitigation=use_mitigation,
                name=job_name,
                description=job_description,
                dry_run=dry_run,
                run_type=RunJobType.DEVICE_RUN.value,
            )
        )
        return job

    @errors.graceful_api_errors_message
    def dry_run(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit] | CircuitModel | list[CircuitModel],
        parameters: list | None = None,
        shots: int = 1000,
        observables: SparsePauliOp | list[SparsePauliOp] | list[list[SparsePauliOp]] | None = None,
        device: DeviceModel | None = None,
        device_id: str | None = None,
        options: dict | None = None,
        use_mitigation: bool = False,
        job_name: str | None = None,
        job_description: str | None = None,
    ) -> JobInsights:
        """
        Submit a dry-run job via the dedicated `/dry-run` endpoint.

        TODO(IMPORTANT): This method is temporary and currently mirrors the `run()` flow.
        It is planned to be redesigned/repopulated to return predictive execution metrics and
        validation data for a target device. Do not treat the current behavior as final.
        """
        self._check_experiment()

        if options is None:
            options = {}

        if device is None:
            if device_id is None:
                raise ValueError("The `device` or `device_id` is required to run the circuits.")
            else:
                self.get_device(device_id=device_id)

        if device is not None:
            device_id = device.id

        logged_circuits = self._prepare_circuits(circuits)

        if job_name is None:
            job_name = job_name_from_circuits(logged_circuits)

        parameters, observables = validate_and_normalize_parameters_and_observables(
            parameters,
            observables,
            len(logged_circuits),
            num_parameters_per_circuit=get_num_parameters_per_circuit(circuits),
        )

        insights = self._client.dry_run(
            data=RunSubmitModel(
                experiment_id=self._experiment.id,
                circuit_ids=[c.id for c in logged_circuits],
                parameters=parameters,
                shots=shots,
                observables=observables,
                device_id=device_id,
                options=options,
                use_mitigation=use_mitigation,
                name=job_name,
                description=job_description,
                dry_run=True,
                run_type=RunJobType.DEVICE_RUN.value,
            )
        )
        return insights

    @errors.graceful_api_errors_message
    def statevector_run(
        self,
        circuits: QuantumCircuit | list[QuantumCircuit] | CircuitModel | list[CircuitModel],
        job_name: str | None = None,
        job_description: str | None = None,
    ) -> RunJobModel:
        """Run quantum circuits on a statevector simulator and obtain exact amplitudes of the wavefunctions.

        This execution type is restricted to non-parametrized circuits up to 20 qubits in size. Circuits may contain
        Haiqu gates, but no mid-circuit measurements or other logical operations. Final measurements in the circuit,
        if present, will be ignored. Statevector is measured over all qubits in their standard qiskit order.

        The returned amplitude ordering matches the indexing convention used by amplitude-based data loaders
        (:meth:`vector_loading`, :meth:`block_vector_loading`, :meth:`function_loading`,
        :meth:`distribution_loading`, :meth:`mps_loading`): index ``i`` corresponds to the computational-basis
        ket whose binary label has rightmost bit = qubit 0.

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

        Examples:
            Single circuit:

            >>> from qiskit import QuantumCircuit
            >>> qc = QuantumCircuit(2)
            >>> qc.h(0)
            >>> qc.cx(0, 1)
            >>> qc.measure_all()  # measurements can be present or not
            >>> job = haiqu.statevector_run(qc)
            >>> job.result()  # Returns: [statevector]
            [array([0.70710678+0.j, 0.        +0.j, 0.        +0.j, 0.70710678+0.j])]

            Multiple circuits:

            >>> from qiskit import QuantumCircuit
            >>> import numpy as np
            >>> qc_bell = QuantumCircuit(2)  # standard bell state
            >>> qc_bell.h(0)
            >>> qc_bell.cx(0, 1)
            >>> bell_gate_phased, _ = haiqu.vector_loading([1, 0, 0, 1.j]).result()  # notice the imaginary amplitude
            >>> qc_bell_phased = QuantumCircuit(2)  # bell state with different phase on |11> state
            >>> qc_bell_phased.compose(bell_gate_phased, inplace=True)
            >>> job = haiqu.statevector_run([qc_bell, qc_bell_phased])
            >>> np.round(job.result(), 3)
            array([[ 0.707+0.j   ,  0.   +0.j   ,  0.   +0.j   ,  0.707+0.j   ],
                   [ 0.707+0.j   ,  0.   +0.j   ,  0.   +0.j   , -0.   +0.707j]])
        """
        self._check_experiment()
        logged_circuits = self._prepare_circuits(circuits)

        if job_name is None:
            job_name = job_name_from_circuits(logged_circuits)

        # Submit the job
        job = self._client.run(
            data=RunSubmitModel(
                experiment_id=self._experiment.id,
                circuit_ids=[c.id for c in logged_circuits],
                name=job_name,
                description=job_description,
                run_type=RunJobType.STATEVECTOR_RUN.value,
            )
        )
        return job

    def build_lr_qaoa_circuit(
        self,
        problem: QUBO,
        p: int = 10,
        initial_state: Optional[QuantumCircuit] = None,
        alphas: Optional[list[float]] = None,
        betas: Optional[list[float]] = None,
        delta: float = 0.5,
        name: Optional[str] = None,
    ) -> CircuitModel:
        """Build an LR-QAOA circuit for a QUBO problem.

        See https://arxiv.org/abs/2405.09169 for more details on LR-QAOA.

        Generates the LR-QAOA circuit on the Haiqu API server and returns it immediately.

        Args:
            problem (QUBO): The QUBO optimization problem.
            p (int): Number of QAOA layers. Defaults to 10.
            initial_state (Optional[QuantumCircuit]): Custom initial state circuit.
                Defaults to None (uniform superposition).
            alphas (Optional[list[float]]): Cost operator parameters.
                Defaults to None (linear ramp).
            betas (Optional[list[float]]): Mixer operator parameters.
                Defaults to None (linear ramp).
            delta (float): Ramp parameter when alphas/betas not specified. Defaults to 0.5.
            name (Optional[str]): Name for the circuit. Defaults to auto-generated name.

        Returns:
            CircuitModel: The generated circuit (ready to use immediately).

        Examples:
            >>> # Build and get circuit immediately
            >>> circuit_model = haiqu.build_lr_qaoa_circuit(problem, p=10)
            >>> print(circuit_model.id)

            >>> # Use in a workflow
            >>> circuit = haiqu.build_lr_qaoa_circuit(problem, p=20, delta=0.3)
            >>> run_job = haiqu.run(circuit, shots=1000)
        """
        from . import schemas

        self._check_experiment()

        # Serialize initial state to QPY if provided
        initial_state_qpy = None
        if initial_state is not None:
            initial_state_qpy = to_qpy(initial_state)

        # Generate default name if not provided
        if name is None:
            name = f"lr-qaoa-p{p}"

        # Create submit model
        submit_data = schemas.LRQAOACircuitSubmitModel(
            experiment_id=self._experiment.id,
            lp_problem=problem.to_lp_string(),
            p=p,
            initial_state_qpy=initial_state_qpy,
            alphas=alphas,
            betas=betas,
            delta=delta,
            name=name,
        )

        # Call API and get circuit immediately (synchronous)
        return self._client.build_lr_qaoa_circuit(data=submit_data)

    @errors.graceful_api_errors_message
    def solve_qubo(
        self,
        problem: "QUBO",
        # Circuit parameters
        p: int = 10,
        initial_state: Optional[QuantumCircuit] = None,
        alphas: Optional[list[float]] = None,
        betas: Optional[list[float]] = None,
        delta: float = 0.5,
        # Execution parameters
        shots: int = 1000,
        device: DeviceModel = None,
        device_id: str = None,
        options: Optional[dict] = None,
        use_packing: bool = False,
        pack_size: Optional[int] = None,
        # Compression parameters
        # TODO: replace compression_options: Optional[dict] with compression_options: Optional[CompressionOptions]
        compression: bool = False,
        compression_options: Optional[dict] = None,
        # Post-processing parameters
        postprocess_iterations: int = 5,
        seed: Optional[int] = None,
        # CVaR parameter
        cvar_alpha: Optional[float] = 0.1,
    ) -> "SolverResult":
        """
        Solve a QUBO optimization problem using Linear Ramp QAOA (LR-QAOA).

        This high-level method orchestrates the complete LR-QAOA workflow:
        1. Builds LR-QAOA circuit with custom parameter schedules
        2. Optionally compresses the circuit
        3. Executes on specified backend
        4. Post-processes results using Haiqu API
        5. Calculates CVaR expectation

        Args:
            problem (QUBO): The QUBO optimization problem to solve.
            p (int): Number of QAOA layers. Defaults to 10.
            initial_state (Optional[QuantumCircuit]): Custom initial state. Defaults to None (uniform superposition).
            alphas (Optional[list[float]]): Cost operator parameters. Defaults to None (linear ramp).
            betas (Optional[list[float]]): Mixer operator parameters. Defaults to None (linear ramp).
            delta (float): Ramp parameter when alphas/betas not specified. Defaults to 0.5.
            shots (int): Number of measurement shots. Defaults to 1000.
            device (DeviceModel): Device to execute on. Defaults to None.
            device_id (str): Id of the device to execute on. Defaults to None.
            options (Optional[dict]): Additional device options.
            use_packing (bool): Whether to use circuit packing for efficient device utilization. Defaults to False.
                **Warning:** Experimental — packing replicates circuits on unused device qubits
                to run multiple copies in parallel, which may increase errors for deeper input circuits.
                For example, a 4-qubit circuit with pack_size=2 and 1000 shots runs two copies
                in parallel with 500 shots each, yielding 1000 shots of results while only paying
                for 500 shot executions on the QPU — a 2x cost saving.
            pack_size (Optional[int]): Number of circuit copies to pack onto the device. Must be >= 2.
                Only valid when ``use_packing=True``. If ``None`` (default), the backend will
                pack into at most 2/3 of the device qubits.
            compression (bool): Whether to compress the circuit. Defaults to False.
            compression_options (Optional[dict]): Compression options dictionary.
                See method `state_compression` for details.
            postprocess_iterations (int): Maximum number of post-processing passes. Defaults to 5.
                Controls the number of complete passes through all bits. Higher values may find
                better local minima but increase runtime. Default of 5 provides good balance
                between speed and solution quality.
            seed (Optional[int]): Random seed for reproducible post-processing results. Defaults to None.
            cvar_alpha (Optional[float]): CVaR alpha parameter. None to disable CVaR. Defaults to 0.1.

        Returns:
            SolverResult: Complete solution with raw/processed results, costs, and metadata.

        Note:
            Bitstrings use Qiskit convention: rightmost bit = qubit 0.
            Example: "101" means qubit 0=1, qubit 1=0, qubit 2=1.

        Examples:
            Basic usage:

            >>> result = haiqu.solve_qubo(problem)

            With more post-processing iterations:

            >>> result = haiqu.solve_qubo(
            ...     problem,
            ...     postprocess_iterations=10,
            ... )

            With custom schedules and compression:

            >>> result = haiqu.solve_qubo(
            ...     problem,
            ...     p=100,
            ...     alphas=my_alphas,
            ...     betas=my_betas,
            ...     compression=True,
            ... )

            Without CVaR:

            >>> result = haiqu.solve_qubo(problem, cvar_alpha=None)
        """

        self._check_experiment()

        # Validate cvar_alpha (not sent to API, used locally)
        # Other parameters validated by Pydantic models (LRQAOACircuitSubmitModel, RunSubmitModel, PostprocessParams)
        if cvar_alpha is not None and not 0 < cvar_alpha <= 1:
            raise ValueError(f"cvar_alpha must be between 0 and 1 (exclusive of 0) or None, got {cvar_alpha}")

        # 1. Build LR-QAOA circuit (synchronous API call)
        # Parameters p, delta, alphas, betas validated by LRQAOACircuitSubmitModel
        lr_circuit_model = self.build_lr_qaoa_circuit(
            problem=problem,
            p=p,
            initial_state=initial_state,
            alphas=alphas,
            betas=betas,
            delta=delta,
            name=f"lr-qaoa-p{p}",
        )
        # lr_circuit_model is a CircuitModel (already saved in DB by API)
        metadata = {"lr_qaoa_circuit_id": lr_circuit_model.id}

        # 2. Optional compression
        compression_quality = None
        compression_info = None
        if compression:
            # Forward user-provided compression options directly; state_compression supplies defaults
            comp_opts = compression_options or {}

            # Deep copy to preserve the parameters for the user
            compression_info = copy.deepcopy(comp_opts)
            compression_job = self.state_compression(lr_circuit_model, **comp_opts)
            compressed_circuit_model = compression_job.result()
            compression_quality = compression_job.quality
            # compressed_circuit_model is already a CircuitModel, no need to log again
            metadata["compressed_circuit_id"] = compressed_circuit_model.id
            metadata["compression_job_id"] = compression_job.id
            circuit_to_run = compressed_circuit_model
        else:
            circuit_to_run = lr_circuit_model

        # 3. Run circuit
        run_job = self.run(
            circuit_to_run,
            shots=shots,
            device=device,
            device_id=device_id,
            options=options or {},
            use_packing=use_packing,
            pack_size=pack_size,
        )
        metadata["job_id"] = run_job.id

        # 4. Get results
        raw_counts = run_job.result()[0]

        # 5. Compute raw costs
        raw_costs = {}
        for bitstring in raw_counts.keys():
            raw_costs[bitstring] = problem.cost(bitstring)

        # 6. Post-process
        processed_costs, processed_counts = self.postprocess(
            counts=raw_counts,
            problem=problem,
            postprocess_iterations=postprocess_iterations,
            seed=seed,
        )

        # 7. Find best solutions
        best_processed = min(processed_costs, key=processed_costs.get)

        # 8. Calculate expectations
        expectation = sum(raw_costs[bs] * prob for bs, prob in raw_counts.items()) / sum(raw_counts.values())

        # Calculate CVaR on post-processed counts for better quality
        cvar = cvar_expectation(
            processed_counts,
            problem,
            alpha=cvar_alpha or 1.0,
        )

        # 9. Return SolverResult
        return SolverResult(
            raw_counts=raw_counts,
            raw_costs=raw_costs,
            processed_counts=processed_counts,
            processed_costs=processed_costs,
            best_solution=best_processed,
            best_cost=processed_costs[best_processed],
            expectation_value=expectation,
            cvar_expectation=cvar,
            cvar_alpha=cvar_alpha or 1.0,
            metadata=metadata,
            compression_quality=compression_quality,
            compression_info=compression_info,
        )

    def postprocess(
        self,
        counts: dict[str, Union[int, float]],
        problem: QUBO,
        postprocess_iterations: int = 5,
        seed: Optional[int] = None,
    ) -> tuple[dict[str, float], dict[str, Union[int, float]]]:
        """
        Apply post-processing to quantum measurement results using Haiqu API.

        This method uses the Haiqu API's optimized postprocessing algorithms to improve
        quantum optimization results. Requires authentication via login().

        Args:
            counts: Dictionary mapping bitstrings to their measurement counts or probabilities.
                Bitstrings must be in Qiskit's little-endian convention (rightmost bit = qubit 0).
            problem: The QUBO problem instance.
            postprocess_iterations: Maximum number of optimization passes. Defaults to 5.
            seed: Random seed for reproducible results. Defaults to None.

        Returns:
            Tuple of (optimized_costs, optimized_counts):
                - optimized_costs: Dict mapping bitstrings to their costs
                - optimized_counts: Dict mapping bitstrings to aggregated counts

        Note:
            Bitstrings use Qiskit convention: rightmost bit = qubit 0.
            Example: "101" means qubit 0=1, qubit 1=0, qubit 2=1.

        Examples:
            Basic usage:

            >>> counts = {"0101": 100, "1010": 50}
            >>> costs, opt_counts = haiqu.postprocess(
            ...     counts=counts,
            ...     problem=my_qubo
            ... )

            With custom parameters:

            >>> costs, opt_counts = haiqu.postprocess(
            ...     counts=counts,
            ...     problem=my_qubo,
            ...     postprocess_iterations=10,
            ...     seed=42
            ... )

        Raises:
            ValueError: If no API client is available (not logged in).
            ValidationError: If parameters have incorrect types.
        """
        # Validate parameters early by constructing PostprocessParams
        # This ensures type checking happens before any other operations
        from .schemas import PostprocessParams

        params = PostprocessParams(
            postprocess_iterations=postprocess_iterations,
            seed=seed,
        )

        if self._client is None:
            raise ValueError(
                "Postprocessing requires API authentication. Please login first:\n\n"
                "  haiqu = Haiqu()\n"
                "  haiqu.login(api_access_key='your-key')\n"
                "  costs, counts = haiqu.postprocess(counts=..., problem=...)"
            )

        return self._client.api_postprocess(
            counts=counts,
            problem=problem,
            postprocess_iterations=params.postprocess_iterations,
            seed=params.seed,
        )

    def postprocess_skqd(
        self,
        results: list[dict[str, float]],
        num_shots: int,
        h1e,
        h2e,
        norb: int,
        nelec: tuple[int, int],
        skqd_options: "SKQDOptions",
    ) -> SKQDJobModel:
        """
        Submit SKQD postprocessing (SQD diagonalization) as an async job.

        Takes the raw probability distributions from all Krylov circuit
        executions and the Hamiltonian tensors, submits them to the Haiqu
        server for classical SQD diagonalization on a background worker,
        and returns a job handle.

        Call ``job.result()`` to block until the worker finishes and get
        an ``SKQDResult``, or ``job.progress()`` for live updates.

        The caller is responsible for passing the correct tensors in the
        same basis used for circuit generation. For example, if circuits
        were built in momentum basis (SIAM), pass momentum-basis tensors.

        Args:
            results: List of probability distribution dicts (bitstring -> float)
                from all Krylov circuit executions, as returned by
                haiqu.run().result().
            num_shots: Number of shots used per circuit execution.
            h1e: One-body Hamiltonian tensor, shape (norb, norb).
                Accepts numpy arrays or nested lists. Must be in the
                same basis as the circuits.
            h2e: Two-body Hamiltonian tensor, shape
                (norb, norb, norb, norb). Accepts numpy arrays or
                nested lists. Must be in the same basis as the circuits.
            norb: Number of spatial orbitals.
            nelec: Tuple of (n_alpha, n_beta) electron counts.
            skqd_options: SKQDOptions with SQD parameters (samples_per_batch,
                num_batches, max_iterations, symmetrize_spin,
                configuration_recovery, seed).

        Returns:
            Job handle. Use `job.result()` to block and get
                `SKQDResult`, or `job.progress()` for live updates.

        Examples:
            >>> from haiqu.sdk.skqd import hubbard_hamiltonian, SKQDOptions, build_hubbard_site_basis_krylov_circuits
            >>> norb = 8
            >>> h1e, h2e = hubbard_hamiltonian(norb, t=1.0, U=8.0)
            >>> circuits = build_hubbard_site_basis_krylov_circuits(norb, krylov_dim=6, dt=0.2, h1e=h1e, h2e=h2e)
            >>> run_job = haiqu.run(circuits, shots=1000, device_id="aer_simulator")
            >>> skqd_job = haiqu.postprocess_skqd(
            ...     results=run_job.result(), num_shots=1000,
            ...     h1e=h1e, h2e=h2e,
            ...     norb=norb, nelec=(4, 4), skqd_options=SKQDOptions(),
            ... )
            >>> result = skqd_job.result()
            >>> print(result.energy)
        """
        if self._client is None:
            raise ValueError(
                "SKQD postprocessing requires API authentication. Please login first:\n\n"
                "  haiqu.login(api_access_key='your-key')"
            )

        if not results:
            raise ValueError("results must not be empty")

        # Validate bitstring lengths: all must equal 2 * norb
        expected_num_bits = 2 * norb
        for i, prob_dist in enumerate(results):
            first_key = next(iter(prob_dist))
            if len(first_key) != expected_num_bits:
                raise ValueError(
                    f"Bitstring length mismatch in results[{i}]: "
                    f"expected {expected_num_bits} (2 * norb={norb}), "
                    f"got {len(first_key)} for '{first_key}'"
                )

        params = PostprocessSKQDParams(
            samples_per_batch=skqd_options.samples_per_batch,
            num_batches=skqd_options.num_batches,
            max_iterations=skqd_options.max_iterations,
            symmetrize_spin=skqd_options.symmetrize_spin,
            configuration_recovery=skqd_options.configuration_recovery,
            seed=skqd_options.seed,
        )

        # Convert numpy arrays to nested lists for JSON serialization
        h1e_list = h1e.tolist() if hasattr(h1e, "tolist") else h1e
        h2e_list = h2e.tolist() if hasattr(h2e, "tolist") else h2e

        submit_data = SKQDSubmitModel(
            experiment_id=self._experiment.id,
            name="SKQD Postprocessing",
            h1e=h1e_list,
            h2e=h2e_list,
            results=results,
            num_shots=num_shots,
            norb=norb,
            nelec=nelec,
            params=params.model_dump(),
        )

        return self._client.api_postprocess_skqd(submit_data)

    def draw(self, circuit: Union[QuantumCircuit, CircuitModel], style: str = ""):
        """Render a quantum circuit. Intended to be used with the Haiqu Lab user interface.

        This function also returns the matplotlib figure object for use in haiqu.log() and other contexts.

        Args:
            circuit (QuantumCircuit | CircuitModel): The quantum circuit to draw.
            style (str): The CSS class to use. The options are:

                         * "" (default): neutral style
                         * "haiqu_blue": Haiqu blue
                         * "haiqu_light": Haiqu light
                         * "haiqu_light2": Haiqu light (alternative)
                         * "haiqu_neutral": Haiqu neutral
                         * "haiqu_orange": Haiqu orange
                         * "haiqu_pink": Haiqu pink
                         * "neon": classic yellow neon
                         * "neutral": Haiqu light grey

        Returns:
            matplotlib.figure.Figure: The rendered circuit figure.

        Examples:
            >>> haiqu.draw(circuit)

        Use the returned figure to log the circuit diagram:

            >>> fig = haiqu.draw(circuit)
            >>> haiqu.log(fig, name="Look at this circuit!")
        """
        from haiqu.sdk.wiz.jupyter import draw_neon_circuit

        if not isinstance(circuit, (QuantumCircuit, CircuitModel)):
            raise TypeError("The `circuit` must be a QuantumCircuit or CircuitModel instance.")

        if isinstance(circuit, CircuitModel):
            if circuit.qpy is not None:
                circuit = from_qpy(circuit.qpy)
            else:
                raise ValueError("This circuit cannot be drawn, as it has no QPY.")

        draw_neon_circuit(circuit=circuit, style=style)
        return circuit.draw(output="mpl", fold=-1, style="bw")

    def _init(
        self,
        experiment_ctx: str,
        experiment_description: str = "",
        log_source_code: bool = False,
    ) -> None:
        """Internal method to set current experiment."""
        self._experiment = self._client.init_experiment(
            data=ExperimentSubmitModel(name=experiment_ctx, description=experiment_description)
        )

        metrics = {}
        notebook_url = detect_notebook()
        if notebook_url is not None:
            metrics["notebook_url"] = notebook_url

        if log_source_code:
            source_path = detect_source_file()
            if source_path is not None:
                content = read_source_file(source_path)
                if content is not None:
                    metrics["source_code"] = content

        if metrics:
            self._log_experiment_metrics(**metrics)

    def _check_login(self):
        """
        Checks the auth status.
        If no - perform a login based on the environment variables.
        """
        if self._client is None:
            self.login(raise_on_error=True)

    @staticmethod
    def _validate_packing(use_packing: bool, pack_size: Optional[int]) -> None:
        """Validate use_packing and pack_size arguments.

        Raises:
            ValueError: If pack_size is set without use_packing=True,
                        or if pack_size is not an integer >= 2.
        """
        if pack_size is not None:
            if not use_packing:
                raise ValueError("pack_size can only be specified when use_packing=True.")
            if not isinstance(pack_size, int) or isinstance(pack_size, bool):
                raise TypeError(f"pack_size must be an integer, got {type(pack_size).__name__}.")
            if pack_size < 2:
                raise ValueError(f"pack_size must be >= 2, got {pack_size}.")

    @staticmethod
    def _normalize_noise_model_option(device_id: str, options: dict) -> None:
        """Validate and serialize noise model option."""
        noise_model = options.get("noise_model", None)

        if noise_model is None:
            return

        if device_id == "aer_simulator":
            from qiskit_aer.noise import NoiseModel

            if isinstance(noise_model, dict):
                return
            if isinstance(noise_model, NoiseModel):
                options["noise_model"] = noise_model.to_dict(serializable=True)
                return
            raise ValueError(
                "`options['noise_model']` for Aer simulator must be either a " "`dict` or a `qiskit_aer.NoiseModel` object."
            )

        if device_id == "ionq_simulator":
            if isinstance(noise_model, str):
                return
            raise ValueError("`options['noise_model']` for IonQ simulator must be a string identifier.")

        raise ValueError(
            "Custom `options['noise_model']` is only supported when "
            "`device_id='aer_simulator'` or `device_id='ionq_simulator'`."
        )

    def _check_experiment(self):
        """
        Checks if an experiment was initialized.
        If no - init with `DEFAULT_EXPERIMENT`
        """
        if self._experiment is None:
            self._init(DEFAULT_EXPERIMENT)

    def _get_or_create_circuit(self, circuit: QuantumCircuit | CircuitModel, **kwargs) -> CircuitModel:
        """Query API for the circuit by hash or log a new one.
        Associate the circuit with the current experiment.

        If circuit is a CircuitModel, it's returned directly without performing a query.

        Args:
            circuit (QuantumCircuit | CircuitModel): The quantum circuit.

        Returns:
            CircuitModel: circuit meta object
        """
        self._check_experiment()

        if isinstance(circuit, CircuitModel):
            return circuit

        try:
            return self._get_circuit(circuit=circuit)
        except CircuitNotRegisteredInExperimentError:
            pass  # If the circuit is not registered, we will log it now.

        for instruction in circuit.data:
            if instruction.label is not None and len(instruction.label) > 65535:
                raise ValueError(
                    (
                        "Circuit contains instruction with label longer than 65,535 characters: "
                        f"'{instruction.operation.name}'."
                        "\nThis could be happen when label is generated from the parameter values with very long string "
                        "representation.\nPlease assign the label explicitly."
                    )
                )

        try:
            qpy_dump = to_qpy(circuit)
        except Exception as e:
            raise ValueError(f"Failed to serialize the circuit to QPY format: {e}") from e

        return self._client.submit_circuit(
            data=CircuitSubmitModel(
                experiment_id=self._experiment.id,
                name=kwargs.get("name", circuit.name),
                qpy_dump=qpy_dump,
                hash=get_circuit_hash(circuit),
                description=kwargs.get("description", ""),
            )
        )

    def _prepare_circuits(
        self,
        circuits: QuantumCircuit | CircuitModel | list[QuantumCircuit | CircuitModel],
    ) -> list[CircuitModel]:
        """Validate and log circuits."""

        if not isinstance(circuits, list):
            circuits = [circuits]

        # Log the circuits if they are not logged yet
        logged_circuits = []

        for c in circuits:
            if isinstance(c, (QuantumCircuit, CircuitModel)):
                c_logged = self._get_or_create_circuit(circuit=c)
                logged_circuits.append(c_logged)
            else:
                raise ValueError("The `circuits` must be a logged circuit or a QuantumCircuit, or a list of these types.")

        return logged_circuits

    def _get_circuit(self, circuit: QuantumCircuit) -> Union[CircuitModel, None]:
        """Query API for the circuit by hash.

        Args:
            circuit (QuantumCircuit): The circuit.
        """
        try:
            c = self._client.get_circuit(
                experiment_id=self._experiment.id,
                circuit_hash=get_circuit_hash(circuit),
            )
            return c
        except HTTPError as e:
            if e.response.status_code == 404:
                raise CircuitNotRegisteredInExperimentError(
                    "This QuantumCircuit wasn't registered in the experiment."
                    "Use `haiqu.log(<experiment name>, circuit)` to associate the circuit with the experiment."
                ) from None
            raise e

    def _get_or_create_job(self, circuit: QuantumCircuit, job_data: object) -> LocalJobModel:
        """Query API for the job (local run results) or log a new one.

        Args:
            circuit (QuantumCircuit): The quantum circuit with metadata.
            job_data (Qiskit results types): The local run results.
        """
        try:
            return self._get_job(job_data=job_data)
        except JobNotRegisteredInExperimentError:
            return self._client.submit_job(
                data=LocalJobSubmitModel(
                    experiment_id=self._experiment.id,
                    hash=str(get_job_hash(job_data=job_data)),
                    circuit_hash=get_circuit_hash(circuit),
                    results=json.dumps(get_job_results(job_data)),
                    device_id=get_job_device(job_data),
                    shots=json.dumps(get_job_shots(job_data)),
                    name=get_job_name(job_data),
                    status=JobStatus.DONE.value,
                    job_type=JobType.LOCAL.value,
                )
            )

    def _get_job(self, job_data: object) -> Union[LocalJobModel, None]:
        """Query API for the job by hash.

        Args:
            job_data (Qiskit results types): The local run results.
        """
        try:
            return self._client.get_job(
                experiment_id=self._experiment.id,
                job_hash=str(get_job_hash(job_data=job_data)),
            )
        except HTTPError as e:
            if e.response.status_code == 404:
                raise JobNotRegisteredInExperimentError(
                    "This run results wasn't registered in the circuit."
                    "The reults from `backend.run()`, job object, sampler or estimator"
                    "could be logged in the context of the circuit. Please use:"
                    "`haiqu.log(circuit, <results or job/sampler/etc>)`"
                ) from None
            raise e

    def _log_experiment_metrics(self, **kwargs) -> None:
        """Push experiment metrics to Haiqu quantum service.

        Args:
            **kwargs (dict): The metrics.
        """
        if kwargs and self._experiment:
            self._client.submit_experiment_metrics(
                experiment_id=self._experiment.id,
                data=SubmitMetricsModel(metrics=preprocess_metrics(**kwargs)),
            )

    def _log_circuit_metrics(self, circuit: QuantumCircuit | CircuitModel, **kwargs) -> None:
        """Push circuit metrics to Haiqu quantum service.

        Args:
            circuit (QuantumCircuit | CircuitModel): The quantum circuit or metadata.
            **kwargs (dict): The metrics.
        """
        if isinstance(circuit, CircuitModel):
            circuit_hash = circuit.hash
        else:
            circuit_hash = get_circuit_hash(circuit)

        if kwargs:
            self._client.submit_circuit_metrics(
                experiment_id=self._experiment.id,
                circuit_hash=circuit_hash,
                data=SubmitMetricsModel(metrics=preprocess_metrics(**kwargs)),
            )

    def _log_job_metrics(self, job_data: object, **kwargs) -> None:
        """Push job metrics to Haiqu quantum service.

        Args:
            job_data (Qiskit results types): The local run results.
            **kwargs (dict): The metrics.
        """
        if kwargs:
            self._client.submit_job_metrics(
                experiment_id=self._experiment.id,
                job_hash=str(get_job_hash(job_data=job_data)),
                data=SubmitMetricsModel(metrics=preprocess_metrics(**kwargs)),
            )

    @staticmethod
    def save_aws_credentials(
        aws_access_key_id: str,
        aws_secret_access_key: str,
        aws_default_region: str = AWS_DEFAULT_REGION,
        aws_session_token: Optional[str] = None,
    ):
        """
        Save AWS credentials in the environment variables:

        - "AWS_ACCESS_KEY_ID"
        - "AWS_SECRET_ACCESS_KEY"
        - "AWS_DEFAULT_REGION"
        - "AWS_SESSION_TOKEN" (optional; required for temporary/STS credentials)

        NOTE: overwrites existing environment variables with the same names.

        Args:
            aws_access_key_id (str): AWS access key ID.
            aws_secret_access_key (str): AWS secret access key.
            aws_default_region (str): AWS default region. Defaults to "us-east-1".
            aws_session_token (Optional[str]): AWS session token for temporary/STS credentials. Defaults to None.
        """
        os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key
        os.environ["AWS_DEFAULT_REGION"] = aws_default_region
        if aws_session_token is not None:
            os.environ["AWS_SESSION_TOKEN"] = aws_session_token

    def _inject_device_credentials(self, device_id: str, credentials: dict) -> None:
        """Inject provider credentials into ``credentials`` based on the device id.

        IQM Resonance devices are matched with ``startswith("iqm")`` — NOT ``"iqm" in ...`` — so
        AWS-hosted IQM devices such as ``aws_iqm_garnet`` route to AWS credentials, not IQM.
        Shared by ``run`` and ``flow`` so the dispatch logic lives in one place.
        """
        if device_id.lower().startswith("aws"):
            self.update_aws_credentials(credentials)
        if device_id.lower().startswith("ibm"):
            self.update_ibm_credentials(credentials)
        if device_id.lower().startswith("iqm"):
            self.update_iqm_credentials(credentials)

    @staticmethod
    def update_aws_credentials(options):
        """
        Inline update options dictionary with AWS credentails values:

        - "aws_access_key_id"
        - "aws_secret_access_key"
        - "aws_default_region"
        - "aws_session_token" (optional; required for temporary/STS credentials)

        Priorities:

        1) Implicit values in "options" dictionary.
        2) Environment variables.
        3) AWS credentials and config files.

        Args:
            options (dict): Backend options dictionary.
        """

        config_path = os.path.expanduser("~/.aws/config")
        credentials_path = os.path.expanduser("~/.aws/credentials")

        aws_config = configparser.ConfigParser()
        aws_config.read([credentials_path, config_path])

        if aws_config.has_section("default"):
            config = aws_config["default"]
        else:
            config = {}

        aws_access_key_id = options.get(
            "aws_access_key_id", os.getenv("AWS_ACCESS_KEY_ID", config.get("aws_access_key_id", None))
        )

        aws_secret_access_key = options.get(
            "aws_secret_access_key", os.getenv("AWS_SECRET_ACCESS_KEY", config.get("aws_secret_access_key", None))
        )

        aws_default_region = options.get(
            "aws_default_region",
            os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", config.get("region", AWS_DEFAULT_REGION))),
        )

        aws_session_token = options.get(
            "aws_session_token",
            os.getenv("AWS_SESSION_TOKEN", config.get("aws_session_token", None)),
        )

        if aws_default_region is None or aws_secret_access_key is None or aws_access_key_id is None:
            raise ValueError(
                "AWS credentials are required to run on AWS Braket devices. "
                "Please provide them in the `options`, save them using "
                "`haiqu.save_aws_credentials()`, or configure them "
                " in the AWS credentials and config files."
            )

        options["aws_access_key_id"] = aws_access_key_id
        options["aws_secret_access_key"] = aws_secret_access_key
        options["aws_default_region"] = aws_default_region
        if aws_session_token is not None:
            options["aws_session_token"] = aws_session_token

    @staticmethod
    def save_ibm_credentials(ibm_quantum_token: str, ibm_quantum_instance: str):
        """
        Save IBM Quantum API token and instance
        using QiskitRuntimeService with the name "haiqu_ibm_account".
        NOTE: overwrites existing account with the name "haiqu_ibm_account".

        Args:
            ibm_quantum_token (str): IBM Quantum API token.
            ibm_quantum_instance (str): IBM Quantum instance name.
        """
        QiskitRuntimeService.save_account(
            token=ibm_quantum_token, instance=ibm_quantum_instance, name="haiqu_ibm_account", overwrite=True
        )

    @staticmethod
    def update_ibm_credentials(options):
        """
        Inline update options dictionary with IBM Quantum API token value "ibm_quantum_token".

        Priorities:

        1) Implicit value in "options" dictionary.
        2) IBM Quantum config file.

        Args:
            options (dict): Options dictionary.
        """
        ibm_token = options.get("ibm_quantum_token", None)
        ibm_instance = options.get("ibm_quantum_instance", None)

        if ibm_token is None or ibm_instance is None:
            accounts = QiskitRuntimeService.saved_accounts()
            if "haiqu_ibm_account" in QiskitRuntimeService.saved_accounts():
                account = accounts["haiqu_ibm_account"]
                if ibm_token is None:
                    ibm_token = account["token"]
                if ibm_instance is None:
                    ibm_instance = account["instance"]
            else:
                raise ValueError(
                    "IBM Quantum API token and instance are required to run on IBM Quantum devices. "
                    "Please provide them in the `options`, save them using "
                    "`haiqu.save_ibm_credentials()`, or using "
                    "`QiskitRuntimeService.save_account()` with the name 'haiqu_ibm_account'"
                )

        options["ibm_quantum_token"] = ibm_token
        options["ibm_quantum_instance"] = ibm_instance

    @staticmethod
    def save_iqm_credentials(iqm_token: str, iqm_server_url: str = IQM_DEFAULT_SERVER_URL):
        """
        Save IQM Resonance credentials in the environment variables:

        - "IQM_TOKEN"
        - "IQM_SERVER_URL"

        NOTE: overwrites existing environment variables with the same names.

        Args:
            iqm_token (str): IQM Resonance API token.
            iqm_server_url (str): IQM Resonance server URL. Defaults to the Resonance production URL.
        """
        os.environ["IQM_TOKEN"] = iqm_token
        os.environ["IQM_SERVER_URL"] = iqm_server_url

    @staticmethod
    def update_iqm_credentials(options):
        """
        Inline update options dictionary with IQM Resonance credential values:

        - "iqm_token"
        - "iqm_server_url"

        Priorities:

        1) Implicit values in "options" dictionary.
        2) Environment variables ("IQM_TOKEN", "IQM_SERVER_URL").

        Args:
            options (dict): Backend options dictionary.
        """
        iqm_token = options.get("iqm_token", os.getenv("IQM_TOKEN", None))
        iqm_server_url = options.get("iqm_server_url", os.getenv("IQM_SERVER_URL", IQM_DEFAULT_SERVER_URL))

        if iqm_token is None:
            raise ValueError(
                "An IQM Resonance API token is required to run on IQM devices. "
                "Please provide it in the `options` as `iqm_token`, or save it using "
                "`haiqu.save_iqm_credentials()`."
            )

        options["iqm_token"] = iqm_token
        options["iqm_server_url"] = iqm_server_url


haiqu = Haiqu()
