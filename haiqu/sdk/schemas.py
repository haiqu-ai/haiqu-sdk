"""
The Haiqu SDK: Schemas and data models used for analytics, data loading and run workflows.
"""

from contextvars import ContextVar
import enum
from datetime import datetime
from functools import cached_property
import time
from typing import Annotated, Any, Dict, Iterable, Optional, Union, List, Tuple, cast
import json
import numpy as np

from pydantic import BaseModel, ConfigDict, TypeAdapter, Field

# from qiskit.transpiler import Target

from . import gates
from . import errors
from . import exceptions
from .errors import JUPYTER_LAB  # TODO: find better place for this constant
from .hybrid import HybridProgram
from .utils import from_qpy, setup_logger
from .qml.optimizer import NFTOptimizerOptions, OptimizerOptionsUnion

JOB_POLL_DELAY = 5
JOB_POLL_MAX_TIMES = 60

CORE_METRICS = (
    "qubits",
    "num_qubits_active",
    "num_parameters",
    "depth",
    "depth_2q",
    "gates_1q",
    "gates_2q",
    "other_ops",
    "other_gates",
    "gates_total",
    "instructions_total",
)
ADVANCED_METRICS = (
    "program_communication",
    "critical_depth",
    "entanglement_ratio",
    "parallelism",
    "liveness",
    "kl_divergence",
)

logger = setup_logger(__name__)


_current_client = ContextVar("_current_client")


class ClientMixin:
    """Mixin for models that need access to an ``ApiClient``."""

    def model_post_init(self, *args, **kwargs) -> None:
        super().model_post_init(*args, **kwargs)
        self._client = _current_client.get(None)


class ParseIterableMixin:
    """Mixin for parsing JSON list with items."""

    @classmethod
    def parse_items(cls, items: Iterable) -> Iterable:
        """Parses JSON list with items"""
        ItemsList = TypeAdapter(Iterable[cls])
        return list(ItemsList.validate_python(items))


class ArtifactModel(BaseModel, ParseIterableMixin):
    """Artifact model."""

    name: str
    artifact_type: str
    artifact_data: Optional[Any] = None
    creation_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None


class ExperimentSubmitModel(BaseModel):
    """Experiment submit payload data model."""

    name: str
    description: Optional[str] = ""
    tags: Optional[str] = ""
    metrics: Optional[dict] = {}


class ExperimentUpdateModel(BaseModel):
    """Experiment update payload data model."""

    name: Optional[str] = None
    description: Optional[str] = None


class UserModel(BaseModel):
    """
    User model.

    MVP: Haiqu user is created by Haiqu team, auth with username/password.
    V2: Haiqu API user is OAuth authenticated user from IBM, Google, etc.
    """

    email: str
    username: str
    first_name: str
    last_name: str
    api_access_key: str

    def full_name(self):
        return f"{self.first_name} {self.last_name}"


class ReferenceModel(BaseModel, ParseIterableMixin):
    """Key metrics data model."""

    name: str
    program_communication: float
    critical_depth: float
    entanglement_ratio: float
    parallelism: float
    liveness: float


class BaseExperimentModel(BaseModel, ParseIterableMixin):
    """
    Base Experiment model.
    It is used as the base for the following classes:
    * ExperimentModel - general model used by the SDK
    * ContextExperimentModel - model with additional context used by the MCP (API)
    """

    id: str
    name: str
    description: str
    creation_date: datetime
    tags: Optional[str] = ""
    circuits_count: Optional[int] = 0
    jobs_count: Optional[int] = 0
    last_action_date: Optional[datetime] = None

    def __repr__(self):
        return f"Experiment {self.name!r}"


class ExperimentModel(BaseExperimentModel):
    """
    Experiment model.

    Haiqu experiment is a parent/organizer entity for circuits and jobs.
    """

    user_id: int
    user: UserModel
    metrics: Optional[dict] = {}


class CircuitSubmitModel(BaseModel):
    """Circuit submit payload data model."""

    experiment_id: str
    name: Optional[str] = ""
    description: Optional[str] = ""
    qpy_dump: str
    hash: str
    tags: Optional[str] = ""
    metrics: Optional[dict] = {}


class SubmitMetricsModel(BaseModel):
    """Circuit metrics submit payload data model."""

    metrics: dict


class CircuitStatus(enum.Enum):
    """Class for circuit analytics calculation job status."""

    SUBMITTED = "Submitted"
    RUNNING = "Running analytics computation"
    CORE_METRICS = "Core metrics computation is done"
    ADVANCED_METRICS = "Advanced metrics computation is done"
    EVOLUTION = "Evolution computation is done"
    DONE = "Done"
    ERROR = "Error"


CIRCUIT_DONE_STATUSES = (
    CircuitStatus.DONE,
    CircuitStatus.ERROR,
)

CORE_METRICS_READY_STATUSES = (
    CircuitStatus.CORE_METRICS,
    CircuitStatus.ADVANCED_METRICS,
    CircuitStatus.EVOLUTION,
    CircuitStatus.DONE,
    CircuitStatus.ERROR,
)

ADVANCED_METRICS_READY_STATUSES = (
    CircuitStatus.ADVANCED_METRICS,
    CircuitStatus.EVOLUTION,
    CircuitStatus.DONE,
    CircuitStatus.ERROR,
)

EVOLUTION_READY_STATUSES = (
    CircuitStatus.EVOLUTION,
    CircuitStatus.DONE,
    CircuitStatus.ERROR,
)


class AnalyticsType(enum.Enum):
    """Class for circuit analytics type."""

    CORE_METRICS = "core"
    ADVANCED_METRICS = "advanced"
    EVOLUTION = "evolution"


class CircuitWidgets(enum.Enum):
    """Class for different Jupyter widgets/plots."""

    DETAILS = "Details"
    RADAR = "Radar Plot"
    DIVERSITY = "Gate Diversity"
    LIVENESS = "Liveness per Qubit"
    CORRELATION = "Correlation Matrix"

    EVOLUTION = "Evolution"
    KL_DIVERGENCE = "KL Divergence"
    REALITY_CHECK = "Reality Check"
    TRANSPILATION_PASSES = "Transpilation Passes"
    QUBITS_MAPPING = "Qubits Mapping"


class CircuitAnalyticsModel(BaseModel):
    """Data model for circuit metrics."""

    # Core metrics
    qubits: int  # TODO: Rename to num_qubits_total
    num_qubits_active: Optional[int] = None
    num_parameters: Optional[int] = None
    depth: int
    depth_2q: int
    gates_1q: int
    gates_2q: int
    other_gates: Optional[int] = None
    gates_total: int
    other_ops: int
    instructions_total: Optional[int] = None
    operations_counts: Optional[dict] = None

    # Advanced metrics
    gate_size_distribution: Optional[dict[int, float]] = None
    gate_diversity: Optional[dict[str, float]] = None
    gate_diversity_basis_gates: Optional[dict[str, float]] = None
    gate_count_distribution: Optional[dict[str, int]] = None
    program_communication: Optional[Union[float, str]] = "N/A"
    critical_depth: Optional[Union[float, str]] = "N/A"
    entanglement_ratio: Optional[Union[float, str]] = "N/A"
    parallelism: Optional[Union[float, str]] = "N/A"
    liveness_per_qubit: Optional[Union[list, str]] = "N/A"
    liveness: Optional[Union[float, str]] = "N/A"
    correlation_matrix: Optional[dict] = None

    # KL divergence between given circuit output distribution and uniform distribution
    kl_divergence: Optional[Union[float, str]] = "N/A"

    # Circuit representation in basis gates (e.g.: "RX", "RY", "RZ", "CX")
    circuit_normalized: Optional[str] = None


class CircuitProxyPerformanceModel(BaseModel):
    """Estimated execution performance for a transpiled circuit."""

    estimated_fidelity: Optional[float] = None
    estimated_survival_rate: Optional[float] = None


class CircuitEvolutionModel(BaseModel):
    """
    Data model for circuit evolution.
    """

    metrics: Union[list, str]
    # For large circuits could be not possible to compute:
    kl_divergence: Optional[Union[list, str]] = None
    reality_check: Optional[Union[list, str]] = None


class BaseCircuitModel(BaseModel, ParseIterableMixin):
    """
    Base Circuit model.
    It is used as the base for the following classes:
    * CircuitModel - general model used by the SDK
    * ContextCircuitModel - model with additional context used by the MCP (API)
    """

    id: str
    hash: str
    name: str
    description: Optional[str] = ""
    creation_date: datetime
    user_id: int
    experiment_id: str
    tags: Optional[str] = ""
    status: CircuitStatus
    generated: bool
    parameters: Optional[dict] = None

    # Analytics data
    analytics: Optional[CircuitAnalyticsModel] = None

    # Transpilation-related properties
    transpilation_target: Optional[str] = None
    transpilation_options: Optional[dict] = None
    transpiled_circuit_ids: Optional[list[str]] = None

    # State compression information
    compressed_circuit_ids: Optional[list[str]] = None

    # Run information
    job_ids: Optional[list[str]] = None

    def __repr__(self):
        return f"Haiqu Circuit {self.name!r}"

    @property
    def num_qubits(self) -> int | None:
        if self.analytics:
            return self.analytics.qubits

    @property
    def depth(self) -> int | None:
        if self.analytics:
            return self.analytics.depth


class CircuitModel(ClientMixin, BaseCircuitModel):
    """
    Haiqu Circuit model for Analytics Workflows.

    Haiqu circuit is a quantum circuit logged in the Haiqu cloud or generated by Haiqu tools
    and enriched with analytics data. It wraps Qiskit QuantumCircuit (``qpy`` property, in the form of QPY dump)
    and provides additional methods for analytics and visualization.

    Cloud-computed structure metrics live on ``analytics`` (for example ``analytics.depth``,
    ``analytics.depth_2q``, ``analytics.gates_2q``); use :meth:`core_metrics` or :meth:`wait_for_analytics`
    if they are not yet populated.
    """

    qpy: Optional[str] = None

    # The evolution of metrics over the circuit slices
    evolution: Optional[CircuitEvolutionModel] = None

    # User-defined metrics or artifacts
    metrics: Optional[dict] = None

    def __getattribute__(self, name):
        """Check the circuit analytics availability in case it is not yet ready or an error status."""
        if name in ("analytics", "evolution") and self.status == CircuitStatus.ERROR:
            raise exceptions.CircuitAnalyticsComputationError("Metrics not available, error during the analytics computation.")
        return super().__getattribute__(name)

    def to_gate(self) -> gates.HaiquCircuitGate:
        """
        Convert the circuit into a :class:`~haiqu.sdk.gates.HaiquCircuitGate`.

        Examples:
            Get a gate from a circuit ID:

            >>> gate = haiqu.get_circuit("circ-12345678-1234-5678-1234-567812345678").to_gate()

            Include the gate in a larger circuit:

            >>> circuit = qiskit.QuantumCircuit(gate.num_qubits + 1)
            >>> circuit.h(range(gate.num_qubits))
            >>> circuit.append(gate, range(gate.num_qubits))
            >>> circuit.cx(range(gate.num_qubits), gate.num_qubits)
            >>> circuit.draw()
                 ┌───┐┌────────────────────────────────────────────────────────────┐
            q_0: ┤ H ├┤0                                                           ├──■────────────
                 ├───┤│                                                            │  │
            q_1: ┤ H ├┤1 Haiqucircuit(circ-12345678-1234-5678-1234-567812345678,3) ├──┼────■───────
                 ├───┤│                                                            │  │    │
            q_2: ┤ H ├┤2                                                           ├──┼────┼────■──
                 └───┘└────────────────────────────────────────────────────────────┘┌─┴─┐┌─┴─┐┌─┴─┐
            q_3: ───────────────────────────────────────────────────────────────────┤ X ├┤ X ├┤ X ├
                                                                                    └───┘└───┘└───┘

            Run the circuit:

            >>> circuit.measure_all()
            >>> haiqu.run(circuit, backend_name="aer_simulator")
        """
        if self.num_qubits is None:
            raise ValueError("Cannot convert to gate: number of gates not available.")

        return gates.HaiquCircuitGate(
            circuit_id=self.id,
            num_qubits=self.num_qubits,
        )

    @errors.graceful_api_errors_message
    def update_from_backend(self):
        """
        Update the circuit's status, analytics, and evolution fields from the backend API.
        """
        response: CircuitModel = self._client.get_circuit(circuit_id=self.id)
        self.status = response.status
        self.analytics = response.analytics
        self.evolution = response.evolution
        self.transpiled_circuit_ids = response.transpiled_circuit_ids

    @errors.graceful_api_errors_message
    def retrieve_status(self) -> CircuitStatus:
        """
        Query backend for the analytics computation status.
        """
        self.update_from_backend()
        return self.status

    @errors.graceful_api_errors_message
    def compute_analytics(self) -> None:
        """
        Explicitly fire the job to compute core analytics on the backend.
        """
        self._client.compute_analytics(circuit_id=self.id)

    @errors.graceful_api_errors_message
    def compute_advanced_metrics(self) -> None:
        """
        Fire the job to compute advanced metrics on the backend. By default, the advanced metrics are not computed
        when the circuit is logged or generated, but it can be triggered with this method.
        """
        if self.status in ADVANCED_METRICS_READY_STATUSES:
            return

        self._client.compute_analytics(
            circuit_id=self.id,
            analytics_type=AnalyticsType.ADVANCED_METRICS,
        )

    @errors.graceful_api_errors_message
    def compute_evolution(self) -> None:
        """
        Fire the job to compute evolution on the backend. By default, the evolution metrics are not computed
        when the circuit is logged or generated, but it can be triggered with this method.
        """
        if self.status in EVOLUTION_READY_STATUSES:
            return

        self._client.compute_analytics(
            circuit_id=self.id,
            analytics_type=AnalyticsType.EVOLUTION,
        )

    def wait_for_analytics(
        self,
        target_status: list = CORE_METRICS_READY_STATUSES,
        widget: bool = True,
        widget_title: str = "ANALYTICS STATUS",
    ):
        """
        Poll the analytics job. Block and wait for the job to complete.
        Warn user in case it takes too long, and raise an exception in script mode.

        Args:
            target_status (list[CircuitStatus]): The status(es) to wait for.
            widget (bool): If ``True`` (default), render the Jupyter widget and return ``None``.
            widget_title (str): The title to display in the widget.
        """
        from haiqu.sdk.wiz.jupyter import job_progress_widget

        def _log_or_widget(widget_logs: str, logger_logs: str):
            if widget:
                job_progress_widget(widget_title, widget_logs)
            else:
                logger.info(logger_logs)

        if self.status in target_status:
            return

        logs = """Polling the API for the status of the analytics computation...\n"""
        _log_or_widget(logs, logs.strip())  # for the first log entry in case of script mode

        attempts = 1

        try:
            while True:
                status = self.retrieve_status()
                if status in target_status:
                    logs += f"\nDONE: {status.value}"
                    _log_or_widget(logs, f"DONE: {status.value}")
                    break
                logs += f"#{attempts} Status: {status.value}\n"
                _log_or_widget(logs, f"#{attempts} Status: {status.value}")

                attempts += 1
                if attempts == JOB_POLL_MAX_TIMES:
                    logs += """\nMaximum number of attempts reached.

For the large circuits, computing metrics (especially the evolution of
the metrics over the circuit) could take some time."""
                    _log_or_widget(logs, "Maximum number of attempts reached.")
                    if not widget:
                        raise exceptions.CircuitAnalyticsComputationError(
                            "Maximum number of attempts reached while waiting for analytics computation."
                        )
                    break
                time.sleep(JOB_POLL_DELAY)
        except KeyboardInterrupt:
            return

    def wait_for_advanced_metrics(
        self,
        widget: bool = True,
        widget_title: str = "ADVANCED ANALYTICS STATUS",
    ):
        """
        Poll the advanced analytics job. Block and wait for the job to complete.

        Args:
            widget (bool): If ``True`` (default), render the list as a Jupyter widget and return ``None``.
            widget_title (str): The title to display in the widget.
        """
        self.wait_for_analytics(
            target_status=ADVANCED_METRICS_READY_STATUSES,
            widget=widget,
            widget_title=widget_title,
        )

    @errors.graceful_api_errors_message
    def all_metrics(
        self,
        help: bool = False,
        tiles_layout: bool = False,
        widget: bool = None,
    ):
        """
        Render the widget displaying the table of all analytics metrics or return the dict with all metrics.

        The basic and advanced analysis reveals insights into critical components of the circuit,
        connectivity, the depth of two-qubit interactions, the degree of entanglement present,
        and the circuit's capacity to execute operations concurrently.

        The default behavior is to auto-detect the environment and render the widget in Jupyter and return the dict
        in scripts, but it can be overridden with ``widget=True/False`` switch.

        Args:
            help (bool): Wherever possible, display a column providing assistance for each metric.
            tiles_layout (bool): The tiles layout render widget with "float" CSS style.
            widget (bool): The switch to force rendering the widget in Jupyter or return a dict.

        Returns:
            dict | None: The dict with all metrics or ``None`` if ``widget=True`` or Jupyter environment.
        """
        from haiqu.sdk.wiz.jupyter import generate_widget_id, loggable_widget, metrics_as_table

        widget = JUPYTER_LAB if widget is None else widget

        self.compute_advanced_metrics()
        self.wait_for_advanced_metrics(widget=widget)
        if not widget:
            return self.analytics.model_dump()

        widget_id = generate_widget_id()

        return loggable_widget(
            self,
            artifact_type="All Metrics",
            html_str=metrics_as_table(
                self,
                help=help,
                tiles_layout=tiles_layout,
                widget_id=widget_id,
            ),
            widget_id=widget_id,
        )

    @errors.graceful_api_errors_message
    def core_metrics(
        self,
        help: bool = False,
        tiles_layout: bool = False,
        widget: bool = None,
    ):
        """
        Render the widget displaying the table of core analytics metrics or return the dict withcore metrics.

        The core analysis provides insights into the high-level structure of the quantum circuit.

        The default behavior is to auto-detect the environment and render the widget in Jupyter and return the dict
        in scripts, but it can be overridden with ``widget=True/False`` switch.

        Args:
            help (bool): Wherever possible, display a column providing assistance for each metric.
            tiles_layout (bool): The tiles layout render widget with "float" CSS style.
            widget (bool): The switch to force rendering the widget in Jupyter or return a dict.

        Returns:
            dict | None: The dict with core metrics or ``None`` if ``widget=True`` or Jupyter environment.
        """
        from haiqu.sdk.wiz.jupyter import generate_widget_id, loggable_widget, metrics_as_table

        widget = JUPYTER_LAB if widget is None else widget

        self.wait_for_analytics(widget=widget)
        if not widget:
            return self.analytics.model_dump(include=CORE_METRICS)

        widget_id = generate_widget_id()

        return loggable_widget(
            self,
            artifact_type="Core Metrics",
            html_str=metrics_as_table(
                self,
                help=help,
                tiles_layout=tiles_layout,
                core_only=True,
                widget_id=widget_id,
            ),
            widget_id=widget_id,
        )

    @errors.graceful_api_errors_message
    def advanced_metrics(
        self,
        help: bool = False,
        tiles_layout: bool = False,
        widget: bool = None,
    ):
        """
        Render the widget displaying the table with advanced quantum circuit metrics or return the dict
        with advanced metrics.

        The advanced analysis reveals insights into critical components of the circuit,
        connectivity, the depth of two-qubit interactions, the degree of entanglement present,
        and the circuit's capacity to execute operations concurrently.

        The default behavior is to auto-detect the environment and render the widget in Jupyter and return the dict
        in scripts, but it can be overridden with ``widget=True/False`` switch.

        Args:
            help (bool): Wherever possible, display a column providing assistance for each metric.
            tiles_layout (bool): The tiles layout render widget with "float" CSS style.
            widget (bool): The switch to force rendering the widget in Jupyter or return a dict.

        Returns:
            dict | None: The dict with advanced metrics or ``None`` if ``widget=True`` or Jupyter environment.
        """
        from haiqu.sdk.wiz.jupyter import generate_widget_id, loggable_widget, metrics_as_table

        widget = JUPYTER_LAB if widget is None else widget

        self.compute_advanced_metrics()
        self.wait_for_advanced_metrics(widget=widget)
        if not widget:
            return self.analytics.model_dump(include=ADVANCED_METRICS)

        widget_id = generate_widget_id()

        return loggable_widget(
            self,
            artifact_type="Advanced Metrics",
            html_str=metrics_as_table(
                self,
                help=help,
                tiles_layout=tiles_layout,
                advanced_only=True,
                widget_id=widget_id,
            ),
            widget_id=widget_id,
        )

    @staticmethod
    def compare_metrics(
        circuits: List["CircuitModel"],
        help: bool = False,
        tiles_layout: bool = False,
    ):
        """
        Render the widget displaying the table comparing core metrics of two circuits.

        Args:
            circuits (list[CircuitModel]): The circuits to compare.
            help (bool): Wherever possible, display a column providing assistance for each metric.
            tiles_layout (bool): The tiles layout render widget with "float" CSS style.

        Returns:
            str: Jupyter Lab Widget.
        """
        if not all(isinstance(circuit, CircuitModel) for circuit in circuits):
            raise ValueError(
                "The circuits must be CircuitModel instances. "
                "You can compare unlogged circuits by using haiqu.compare_metrics(*circuits)."
            )

        from haiqu.sdk.wiz.jupyter import generate_widget_id, loggable_widget, compare_metrics_as_table

        if not circuits:
            raise ValueError("At least one circuit is required.")

        widget_id = generate_widget_id()

        return loggable_widget(
            circuits[0],
            artifact_type="Circuit Comparison",
            html_str=compare_metrics_as_table(
                circuits,
                help=help,
                tiles_layout=tiles_layout,
                widget_id=widget_id,
            ),
            widget_id=widget_id,
            multiple_circuits=True,
        )

    @errors.graceful_api_errors_message
    def draw_radar(
        self,
        help: bool = False,
        tiles_layout: bool = False,
    ):
        """
        Render widget - the radar (wind-rose) plot.

        Args:
            help (bool): Wherever possible, display a column providing assistance for each metric.
            tiles_layout (bool): The tiles layout render widget with "float" CSS style.

        Returns:
            str: Jupyter Lab Widget.
        """
        from IPython.display import display, HTML
        from haiqu.sdk.wiz.jupyter import plot_radar

        self.compute_advanced_metrics()
        self.wait_for_advanced_metrics()
        return display(HTML(plot_radar(self, help=help, tiles_layout=tiles_layout)))

    @errors.graceful_api_errors_message
    def draw_gate_diversity(
        self,
        help: bool = False,
        tiles_layout: bool = False,
        basis_gates: bool = False,
    ):
        """
        Render widget - displays the quantity of different gates in the circuit.

        It has two modes: original and normalized to basis gates.

        Args:
            help (bool): Wherever possible, display a column providing assistance for each metric.
            tiles_layout (bool): The tiles layout render widget with "float" CSS style.
            basis_gates (bool): Render the normalized gates data.

        Returns:
            str: Jupyter Lab Widget.
        """
        from IPython.display import display, HTML
        from haiqu.sdk.wiz.jupyter import plot_gate_diversity

        self.compute_advanced_metrics()
        self.wait_for_advanced_metrics()
        return display(
            HTML(
                plot_gate_diversity(
                    self,
                    help=help,
                    tiles_layout=tiles_layout,
                    basis_gates=basis_gates,
                )
            )
        )

    @errors.graceful_api_errors_message
    def draw_liveness_per_qubit(
        self,
        help: bool = False,
        tiles_layout: bool = False,
    ):
        """
        Render the widget that displays the liveness per qubit plot.

        The liveness metrics indicate the frequency of qubit activity during
        execution relative to the total circuit depth, thereby reflecting qubit utilization.

        Args:
            help (bool): Wherever possible, display a column providing assistance for each metric.
            tiles_layout (bool): The tiles layout render widget with "float" CSS style.

        Returns:
            str: Jupyter Lab Widget.
        """
        from IPython.display import display, HTML
        from haiqu.sdk.wiz.jupyter import plot_liveness_per_qubit

        self.compute_advanced_metrics()
        self.wait_for_advanced_metrics()
        return display(HTML(plot_liveness_per_qubit(self, help=help, tiles_layout=tiles_layout)))

    @errors.graceful_api_errors_message
    def draw_correlation_matrix(
        self,
        help: bool = False,
        tiles_layout: bool = False,
    ):
        """
        Render a heatmap of qubits correlation, displaying interactions between qubits
        to visualize entanglement and connectivity.

        This helps optimize qubit placement on hardware.

        Args:
            help (bool): Wherever possible, display a column providing assistance for each metric.
            tiles_layout (bool): The tiles layout render widget with "float" CSS style.

        Returns:
            str: Jupyter Lab Widget.
        """
        from IPython.display import display, HTML
        from haiqu.sdk.wiz.jupyter import plot_correlation_matrix

        self.compute_advanced_metrics()
        self.wait_for_advanced_metrics()
        return display(HTML(plot_correlation_matrix(self, help=help, tiles_layout=tiles_layout)))

    @errors.graceful_api_errors_message
    def draw(self, style: str = ""):
        """Render a quantum circuit the cool way — neon, Japan 80s style.

        Args:
            style (str): The CSS class to use. The options are:

                         * "" (default): classic neutral style
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
        """
        from haiqu.sdk.wiz.jupyter import draw_neon_circuit

        if self.qpy is not None:
            circuit = from_qpy(self.qpy)
        else:
            raise ValueError("This circuit cannot be drawn, as it has no QPY.")

        draw_neon_circuit(circuit=circuit, style=style)
        return circuit.draw(output="mpl", fold=-1, style="bw")

    @errors.graceful_api_errors_message
    def draw_evolution(
        self,
        metric: str = "depth",
        help: bool = False,
        tiles_layout: bool = False,
    ):
        """
        Generate a plot showing the evolution of key circuit metrics.

        The original circuit is divided gate-by-gate or by some step,
        depending on the initial depth. Metrics are then calculated for each slice.
        This approach provides insights into how critical components of the circuit
        change over time, including connectivity, the depth of two-qubit interactions,
        the degree of entanglement, and the circuit's ability
        to perform operations concurrently.

        Args:
            metric (str): metric according to which to plot the evolution. Possible values are `depth`,
                          `gates_1q`, `gates_2q`, `gates_total`. Defaults to `depth`.
            help (bool): Wherever possible, display a column providing assistance for each metric.
            tiles_layout (bool): The tiles layout render widget with "float" CSS style.

        Returns:
            str: Jupyter Lab Widget.
        """
        from haiqu.sdk.wiz.jupyter import plot_circuit_evolution

        self.compute_evolution()
        self.wait_for_analytics(target_status=EVOLUTION_READY_STATUSES)
        return plot_circuit_evolution(
            circuit=self,
            metric=metric,
            help=help,
            tiles_layout=tiles_layout,
        )

    # -- JUPYTER WIDGETS ------------------------------------------------------
    @cached_property
    def benchmarks(self) -> Iterable:
        """Display circuit key metrics alongside with our pre-run benchmarks.
        API REST `help_benchmarks` queried for the benchmarks.

        Returns:
            Iterable: The benchmarks list with current circuit.
        """

        items = self._client.help_benchmarks()
        items.insert(
            0,
            ReferenceModel(
                name=self.name or self.id,
                program_communication=self.analytics.program_communication,
                critical_depth=self.analytics.critical_depth,
                entanglement_ratio=self.analytics.entanglement_ratio,
                parallelism=self.analytics.parallelism,
                liveness=self.analytics.liveness,
            ),
        )
        return items

    def transpilation_passes(self, device_id):
        """Jupyter widget: table with transpilation passes. Lazy import."""
        from haiqu.sdk.wiz.jupyter import transpilation_passes_as_table

        return transpilation_passes_as_table(self, device_id)


class JobType(enum.Enum):
    """Class for job types."""

    LOCAL = "User local job"
    ANALYTICS = "Analytics"  # circuit.analytics & circuit.evolution
    DEVICE_ANALYTICS = "Device specific analytics"  # circuit.transpilation
    DATA_LOADING = "Data Loading"
    HYBRID = "Hybrid"
    RUN = "Run"
    COMPRESSION = "State Compression"
    TRANSPILATION = "Transpilation"
    VARIATIONAL = "Variational"
    PRETRAINING = "Pretraining"
    SKQD = "SKQD"


class DataLoadingType(enum.Enum):
    """Class for `dl_type`."""

    DISTRIBUTION_LOADING = "DistributionLoading"
    VECTOR_LOADING = "VectorLoading"
    BLOCK_VECTOR_LOADING = "BlockVectorLoading"
    ENTANGLED_MANIFOLD_EMBEDDING = "EntangledManifoldEmbedding"
    MPS_LOADING = "MpsLoading"
    FUNCTION_LOADING = "FunctionLoading"


class RunJobType(enum.Enum):
    """Class for `run_type`."""

    DEVICE_RUN = "Run"  # TODO: consider splitting into QPU run and fake device/simulator run
    STATEVECTOR_RUN = "StatevectorRun"


class CompressionJobType(enum.Enum):
    """Class for `compression_type`."""

    STATE_COMPRESSION = "StateCompression"
    STATE_COMPRESSION_2D = "StateCompression2D"
    SU2_EQUIVARIANT_COMPILATION = "Su2EquivariantCompilation"


class JobStatus(enum.Enum):
    """Class for job status."""

    SUBMITTED = "Submitted"
    INITIALIZING = "Initializing"
    QUEUED = "Queued"
    VALIDATING = "Validating"
    RUNNING = "Running"
    CANCELLED = "Cancelled"
    DONE = "Done"
    ERROR = "Error"


UNFINISHED_JOB_STATUSES = set(
    [
        JobStatus.SUBMITTED,
        JobStatus.INITIALIZING,
        JobStatus.QUEUED,
        JobStatus.VALIDATING,
        JobStatus.RUNNING,
    ]
)

FINISHED_JOB_STATUSES = set(
    [
        JobStatus.DONE,
        JobStatus.ERROR,
        JobStatus.CANCELLED,
    ]
)

assert not (UNFINISHED_JOB_STATUSES & FINISHED_JOB_STATUSES), "Invalid job statuses"
assert set(JobStatus) == UNFINISHED_JOB_STATUSES | FINISHED_JOB_STATUSES, "Invalid job statuses"


class JobStatusModel(BaseModel):
    status: JobStatus


class BaseJobModel(ClientMixin, ParseIterableMixin, BaseModel):
    """
    Base schema for all jobs: data loading, execution, analytics,
    logged local jobs, etc.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: Optional[str] = None
    description: Optional[str] = None
    user_id: int
    experiment_id: str
    status: JobStatus
    job_type: JobType
    device_id: Optional[str] = "Haiqu OS"
    creation_date: datetime
    run_date: Optional[datetime] = None
    finish_date: Optional[datetime] = None
    logs: Optional[str] = None
    quality: Optional[float] = None
    info: Optional[dict] = None
    time: Optional[float] = None
    parameters: Optional[dict | list] = None

    def __repr__(self):
        if errors.JUPYTER_LAB:
            return f"""{self.__class__.__name__} {repr(self.name or self.id)}
Run haiqu.list_jobs() to check its progress and access it by ID as haiqu.get_job("{self.id}").
Check further progress via: job.progress(), job.result()"""
        else:
            return f"{self.__class__.__name__} {repr(self.name or self.id)}"

    def _progress(self, widget_title: str):
        """
        Display the progress widget, stream logs from the job.
        """
        from haiqu.sdk.wiz.jupyter import job_progress_widget

        if self.status not in FINISHED_JOB_STATUSES:
            try:
                while True:
                    job_progress_widget(widget_title, self.logs)
                    if self.retrieve_status() in FINISHED_JOB_STATUSES:
                        break
                    time.sleep(JOB_POLL_DELAY)
            except KeyboardInterrupt:
                return

        job_progress_widget(widget_title, self.logs)

    def _retrieve_status(self) -> "BaseJobModel":
        """
        Query backend for the job status and update fields shared by all job models.
        """
        response: BaseJobModel = self._client.get_job(job_id=self.id)
        self.status = response.status
        self.logs = response.logs
        self.info = response.info
        self.time = response.time
        self.quality = response.quality
        return response

    def cancel(self) -> bool:
        """
        Cancel the job. Returns True if the job was successfully cancelled.
        """
        updated_job = self._client.cancel_job(job_id=self.id)
        self.status = updated_job.status
        self.logs = updated_job.logs
        return self.status == JobStatus.CANCELLED

    def restart(self) -> bool:
        """
        Restart the job. Returns True if the job was successfully restarted.
        """
        updated_job = self._client.restart_job(job_id=self.id)
        self.status = updated_job.status
        self.logs = updated_job.logs
        return self.status == JobStatus.SUBMITTED

    def draw(self, **kwargs):
        """Drawing is only supported for run jobs. Use a RunJobModel instance."""
        raise NotImplementedError("Drawing is only supported for run jobs.")

    def result(self):
        """Baseline method for fetching results for a job.
        Waits till job is finished.
        Raises an exception if result is not available otherwise returns None.

        Returns:
            None
        """
        if self.status not in FINISHED_JOB_STATUSES:
            try:
                while True:
                    if self.retrieve_status() in FINISHED_JOB_STATUSES:
                        break
                    time.sleep(JOB_POLL_DELAY)
            except KeyboardInterrupt:
                return

        if self.status == JobStatus.ERROR:
            raise Exception(f"Job has failed. Job runtime errors are present in logs:\n{self.logs}")
        elif self.status == JobStatus.CANCELLED:
            raise Exception("Job was cancelled, result is not available")
        elif self.status == JobStatus.DONE:
            return  # job was successfully finished
        else:
            raise NotImplementedError(f"Unexpected job status, when retrieving results: {self.status}")


class AnalyticsJobModel(BaseJobModel):
    """
    Analytics job
    """

    circuit_id: str
    analytics_type: AnalyticsType


class LocalJobModel(BaseJobModel):
    """
    Local user-run Job (regular Qiskit job).
    Flow:
        * user run normal Qiskit job
        * they use `haiqu.log(<job|result>)` to send it to Haiqu cloud for analysis.

    Qiskit and IBM runtime has several types of jobs and types of results.
    We do our best to analyze them all.
    """

    hash: str  # hash of the job/result used to prevent double-analyzing same job/result
    circuit_id: Optional[str] = ""
    backend_name: str
    results: Optional[str] = ""
    shot_local: Optional[str] = ""

    # User-defined metrics logged along with job
    metrics: Optional[dict] = {}


class DataLoadingSubmitModel(BaseModel):
    """Data Loading payload data model."""

    experiment_id: Optional[str] = ""  # not used in data_loading_estimates()
    name: Optional[str] = ""  # not used in data_loading_estimates()
    description: Optional[str] = ""
    parameters: dict
    num_qubits: Optional[int] = None  # not used in vector/samples loading
    distribution_name: Optional[str] = None  # used only in distribution loading
    dl_type: str = ""  # loading request


class DataLoadingJobModel(BaseJobModel):
    """
    Data Loading Job (running on Haiqu backend).
    """

    # Distribution parameters
    num_qubits: Optional[int]
    distribution_name: Optional[str] = None
    parameters: dict
    dl_type: DataLoadingType

    # Populated after job finishes
    circuit_id: Optional[str] = None
    fidelity: Optional[float] = None

    def retrieve_status(self) -> JobStatus:
        """
        Query backend for the job status.
        """
        response = cast(DataLoadingJobModel, self._retrieve_status())
        self.circuit_id = response.circuit_id
        self.fidelity = response.fidelity
        self.num_qubits = response.num_qubits

        return self.status

    def progress(self):
        """
        Display the progress widget, stream logs from the job.
        """
        # split the DL type by upper case words, adds a space between them
        dl_type_words = "".join([(" " + s if s.isupper() else s) for s in self.dl_type.value]).strip()
        dl_type_words = dl_type_words.upper()
        widget_title = f"{dl_type_words} JOB PROGRESS"
        self._progress(widget_title)

    def result(self) -> gates.HaiquCircuitGate | None:
        """
        Return job result - HaiquCircuitGate or None if interrupted.
        Block and wait for the job to complete.

        Job's `info` field contains additional information (not every field may be present):
            * fidelity - (global) quantum state fidelity between a state, produced by the data loading circuit, and the
                         desired ideal state
            * num_blocks - equal to the number of blocks in the `block_vector_loading`
            * num_qubits_per_block - number of qubits each encoding block takes in the `block_vector_loading`
            * block_norms - norms of the encoded blocks in the `block_vector_loading`
            * fidelity_per_block - quantum state fidelity of the each block's data loading in the `block_vector_loading`
            * mean_fidelity - average quantum state fidelity over all block's data loading in the `block_vector_loading`
            * global_fidelity - global quantum state fidelity of the `block_vector_loading`
        """
        super().result()

        return gates.HaiquCircuitGate(circuit_id=self.circuit_id, num_qubits=self.num_qubits)


class DataLoadingEstimatesModel(BaseModel):
    estimated_time: float
    estimated_cost: float

    def draw(self, help: bool = False):
        """
        Render widget with the estimates for Haiqu/Jupyter Lab.

        Returns:
            str: Jupyter Lab Widget.
        """
        from haiqu.sdk.wiz.jupyter import draw_data_loading_estimates

        return draw_data_loading_estimates(data=self, help=help)


class StateCompressionEstimatesSubmitModel(BaseModel):
    """
    Data models to use in the Compression estimates request.
    """

    num_qubits: int
    parameters: dict


class StateCompressionEstimatesModel(DataLoadingEstimatesModel):
    """
    Data models to use in the Compression estimates response.
    """

    def draw(self, help: bool = False):
        """
        Render widget with the estimates for Haiqu/Jupyter Lab.

        Returns:
            str: Jupyter Lab Widget.
        """
        from haiqu.sdk.wiz.jupyter import compression_estimates

        return compression_estimates(data=self, help=help)


class HybridSubmitModel(BaseModel):
    """Hybrid submit payload data model."""

    experiment_id: str

    name: Optional[str] = ""
    description: Optional[str] = ""

    program: HybridProgram

    circuit_ids: Annotated[list[str], Field(min_length=1)]
    shots: Annotated[int, Field(ge=1)] = 1000
    parameters: Optional[list] = None
    observables: Optional[List[List[Tuple[List[str], List[float]]]]] = None

    device_credentials: dict = {}
    dry_run: bool = False


class HybridJobModel(BaseJobModel):
    """
    Hybrid Job (running on Haiqu backend).
    """

    program: HybridProgram

    input_circuit_ids: list[str]
    shots: int
    parameters: Optional[list] = None
    observables: Optional[List[List[Tuple[List[str], List[float]]]]] = None

    # device_credentials omitted
    dry_run: bool

    # Populated after job finishes
    quantum_results: Optional[list] = None
    estimated_qpu_cost_: Annotated[Optional[dict], Field(alias="estimated_qpu_cost")] = None

    def retrieve_status(self) -> JobStatus:
        """
        Query backend for the job status.
        """
        response = cast(HybridJobModel, self._retrieve_status())
        self.quantum_results = response.quantum_results
        self.estimated_qpu_cost_ = response.estimated_qpu_cost_

        return self.status

    def progress(self):
        """
        Display the progress widget, stream logs from the job.
        """
        self._progress("HYBRID JOB PROGRESS")

    def result(self) -> Union[list, None]:
        """
        Block and wait for the job to complete, then return the results.

        Raises an exception if the job failed, with the logs attached.

        Returns:
            list: Results of the job unless interrupted or failed.
        """
        super().result()
        return self.quantum_results

    @property
    def estimated_qpu_cost(self) -> Optional[dict]:
        """
        Return the estimated QPU cost for the job, based on circuit depth,
        shot count, and device rep delay.

        Returns:
            dict or None: ``{"native": {"amount": <seconds>, "unit": "s"},
            "converted": {"amount": <dollars>, "unit": "USD"}}``

        Warning:
            This property blocks the current CPU thread until the job reaches a
            terminal state (Done/Error), similarly to ``result()``. Use ``retrieve_status()``
            to first check the state of the job to omit the prolonged thread blockade.
        """
        self.result()

        return self.estimated_qpu_cost_

    @property
    def qpu_cost(self) -> Optional[dict]:
        """
        Actual QPU cost after execution. Pricing varies by vendor
        (e.g. time-based for IBM, shot-based for AWS).
        Returns None if there is no QPU cost (e.g. simulator execution).

        Returns:
            dict or None: ``{"native": {"amount": <float>, "unit": "<vendor_unit>"},
            "converted": {"amount": <dollars>, "unit": "USD"}}``

        Warning:
            This property blocks the current CPU thread until the job reaches a
            terminal state (Done/Error), similarly to ``result()``. Use ``retrieve_status()``
            to first check the state of the job to omit the prolonged thread blockade.
        """
        self.result()

        if self.info is None:
            return None
        return self.info.get("qpu_cost")

    @property
    def pre_device_pipeline_time(self) -> Optional[float]:
        """
        Wall-clock seconds to run hybrid program layers up to (but not including)
        the device layer.

        Covers classical work in those layers, such as transpilation, mitigation
        ordering, and observable processing. Populated for both full runs and
        ``dry_run=True`` jobs.

        Returns:
            float or None: Pre-device hybrid pipeline duration in seconds.

        Warning:
            This property blocks the current CPU thread until the job reaches a
            terminal state (Done/Error), similarly to ``result()``. Use ``retrieve_status()``
            to first check the state of the job to omit the prolonged thread blockade.
        """
        self.result()

        if self.info is None:
            return None
        return self.info.get("pre_device_pipeline_time")


class RunSubmitModel(BaseModel):
    """Run submit payload data model."""

    experiment_id: str
    circuit_ids: Optional[list] = []
    parameters: Optional[list] = None
    shots: int = Field(ge=1, description="Number of measurement shots (must be >= 1)", default=1024)
    observables: Optional[List[List[Tuple[List[str], List[float]]]]] = None
    device_id: Optional[str] = ""
    options: Optional[dict] = {}
    use_mitigation: Optional[bool] = False
    name: Optional[str] = ""
    description: Optional[str] = ""
    dry_run: Optional[bool] = False
    run_type: Optional[str] = RunJobType.DEVICE_RUN.value


class RunJobModel(BaseJobModel):
    """
    Run Job (running on Haiqu backend).
    """

    # Run parameters
    circuit_ids: list
    parameters: Optional[list] = None
    shots: int
    observables: Optional[List[List[Tuple[List[str], List[float]]]]] = None
    device_id: str
    options: dict
    use_mitigation: Optional[bool]
    dry_run: bool
    run_type: RunJobType

    # Populated after job finishes
    quantum_results: Optional[list] = None
    estimated_qpu_cost_: Optional[dict] = Field(default=None, alias="estimated_qpu_cost")

    def retrieve_status(self) -> JobStatus:
        """
        Query backend for the job status.
        """
        response = cast(RunJobModel, self._retrieve_status())
        self.quantum_results = response.quantum_results
        self.estimated_qpu_cost_ = response.estimated_qpu_cost_

        return self.status

    def progress(self):
        """
        Display the progress widget, stream logs from the job.
        """
        run_type_words = "".join([(" " + s if s.isupper() else s) for s in self.run_type.value]).strip()
        run_type_words = run_type_words.upper()
        widget_title = f"{run_type_words} JOB PROGRESS"
        self._progress(widget_title)

    def result(self) -> Union[list, None]:
        """
        Return job result - results or None if interrupted.
        Block and wait for the job to complete.
        Raises an exception if the job failed with logs attached.
        Returns:
            list: Results of the job unless interrupted or failed.
            - Without observables: 2D list of quasi-probabilities [[dist1, dist2, ...], [dist1, dist2, ...]]
            - With observables:
                * With parameters: 3D list [circuit][observable][parameter]
                * Without parameters: 2D list [circuit][observable]
            - Exact statevectors for `statevector_run` execution

        Job's `info` field contains additional information (not every field may be present):
            * uncertainty - list of uncertainties for execution with observables
        """
        super().result()

        if self.run_type == RunJobType.DEVICE_RUN:
            return self.quantum_results
        elif self.run_type == RunJobType.STATEVECTOR_RUN:
            return [np.array(sv_real) + 1.0j * np.array(sv_imag) for (sv_real, sv_imag) in self.quantum_results]
        else:
            raise ValueError(f"Unrecognized run type: {self.run_type}")

    @property
    def estimated_qpu_cost(self) -> Optional[dict]:
        """
        Return the estimated QPU cost for the job, based on circuit depth,
        shot count, and device rep delay.

        Returns:
            dict or None: ``{"native": {"amount": <seconds>, "unit": "s"},
            "converted": {"amount": <dollars>, "unit": "USD"}}``

        Warning:
            This property blocks the current CPU thread until the job reaches a
            terminal state (Done/Error), similarly to ``result()``. Use ``retrieve_status()``
            to first check the state of the job to omit the prolonged thread blockade.
        """
        self.result()

        return self.estimated_qpu_cost_

    @property
    def qpu_cost(self) -> Optional[dict]:
        """
        Actual QPU cost after execution. Pricing varies by vendor
        (e.g. time-based for IBM, shot-based for AWS).
        Returns None if there is no QPU cost (e.g. simulator execution).

        Returns:
            dict or None: ``{"native": {"amount": <float>, "unit": "<vendor_unit>"},
            "converted": {"amount": <dollars>, "unit": "USD"}}``

        Warning:
            This property blocks the current CPU thread until the job reaches a
            terminal state (Done/Error), similarly to ``result()``. Use ``retrieve_status()``
            to first check the state of the job to omit the prolonged thread blockade.
        """
        self.result()

        if self.info is None:
            return None
        return self.info.get("qpu_cost")

    def draw(self, help: bool = False):
        """
        Render the execution flow graph for this run job.

        Shows the processing pipeline from input to the target device,
        including packing, mitigation, transpilation, and observable stages where applicable.

        Args:
            help (bool): If ``True``, add a legend explaining each stage. Defaults to ``False``.

        Returns:
            IPython display object with the rendered widget.
        """
        from haiqu.sdk.wiz.jupyter import draw_run_job

        options = self.options or {}

        skip_transpilation = options.get("skip_transpilation", False)
        if skip_transpilation:
            uses_transpilation = False
        else:
            circuit_models = [self._client.get_circuit(circuit_id=cid) for cid in self.circuit_ids]
            all_pre_transpiled = bool(circuit_models) and all(cm.transpilation_target is not None for cm in circuit_models)
            uses_transpilation = not all_pre_transpiled

        uses_observables = bool(self.observables)
        uses_mitigation = bool(self.use_mitigation)
        uses_packing = bool(options.get("use_packing", False))
        emo = options.get("error_mitigation_options") or {}

        try:
            device_label = self._client.get_device(self.device_id).name
        except Exception:
            device_label = self.device_id

        return draw_run_job(
            device_id=device_label,
            uses_observables=uses_observables,
            uses_mitigation=uses_mitigation,
            uses_packing=uses_packing,
            uses_transpilation=uses_transpilation,
            use_advanced=emo.get("advanced_mitigation", True),
            use_noise_tailoring=emo.get("noise_tailoring", False),
            use_dd=emo.get("dynamical_decoupling", True),
            use_readout=emo.get("readout_mitigation", True),
            help=help,
        )

    def to_json(self, path: Optional[str] = None):
        """
        Save job results to a JSON file."""

        if path is None:
            path = f"results_{self.id}.json"

        data = {
            "job_id": self.id,
            "status": self.status.value,
            "logs": self.logs,
            "results": self.quantum_results,
        }

        with open(path, "w") as f:
            json.dump(data, f)


class JobInsights(BaseModel):
    """Dry-run insights payload."""

    job: RunJobModel
    metrics: BaseModel | dict[str, Any] | None = None
    data: BaseModel | dict[str, Any] | None = None


class LocalJobSubmitModel(BaseModel):
    """Job submit data model for user-owned, local job/results.

    * User local Qiskit job/results, haiqu.log(job|results)
    * Data loading job
    * Execution job
    """

    experiment_id: str
    hash: Optional[str] = ""
    circuit_hash: Optional[str] = ""
    results: Optional[str] = ""
    device_id: Optional[str] = ""
    shots: Optional[str] = ""
    name: Optional[str] = ""
    status: str = ""
    job_type: str = ""
    metrics: Optional[dict] = {}


class StateCompressionSubmitModel(BaseModel):
    """Compression submit model."""

    experiment_id: str
    circuit_ids: Optional[list] = []  # id for circuit(s)
    compression_type: str = CompressionJobType.STATE_COMPRESSION.value
    parameters: dict


class StateCompressionJobModel(BaseJobModel):
    """Job returned for state compression function."""

    parameters: dict
    # Populated after job finishes
    circuit_id: Optional[str] = None
    quality: Optional[float] = None
    compression_type: CompressionJobType

    def retrieve_status(self) -> JobStatus:
        """
        Query backend for the job status.
        """
        response = cast(StateCompressionJobModel, self._retrieve_status())
        self.circuit_id = response.circuit_id

        return self.status

    def result(self) -> CircuitModel:
        """
        Return job result - compressed circuit.
        Block and wait for the job to complete.

        Job's `info` field contains additional information (not every field may be present):
            * compression_quality - quality of the compressed circuit
            * success - boolean which signals whether the compression was successful, that is input circuit was reduced
            * compression_status - status (last stage) of the compression process
            * compression_time - wall-clock time taken for the compression
            * compression_percent - percent of reduction of CNOT gates in the input circuit
            * approximation_level - approximation level which was used in the compression
        """
        super().result()

        circuit = self._client.get_circuit(circuit_id=self.circuit_id)
        return circuit

    def progress(self):
        """
        Display the progress widget, stream logs from the job.
        """
        title = "STATE COMPRESSION"
        if self.compression_type.value == CompressionJobType.STATE_COMPRESSION_2D.value:
            title = "STATE COMPRESSION 2D"
        self._progress(f"{title} JOB PROGRESS")


class Su2EquivariantCompilationJobModel(BaseJobModel):
    """Job returned for SU(2)-equivariant gate compilation.

    Mirrors the state-compression surface: ``result()`` returns the
    compressed circuit and ``fidelity`` is the achieved process fidelity of
    the compressed brick to the target unitary. The fit runs server-side;
    only the circuit and the fidelity are exposed.
    """

    parameters: dict
    # Populated after job finishes
    circuit_id: Optional[str] = None
    compression_type: CompressionJobType

    def retrieve_status(self) -> JobStatus:
        """
        Query backend for the job status.
        """
        response = cast(Su2EquivariantCompilationJobModel, self._retrieve_status())
        self.circuit_id = response.circuit_id

        return self.status

    @property
    def fidelity(self) -> Optional[float]:
        """Process fidelity of the compressed brick to the target unitary."""
        return self.quality

    def result(self) -> CircuitModel:
        """
        Return job result - the compressed circuit.
        Block and wait for the job to complete.
        """
        super().result()

        circuit = self._client.get_circuit(circuit_id=self.circuit_id)
        return circuit

    def progress(self):
        """
        Display the progress widget, stream logs from the job.
        """
        self._progress("EQUIVARIANT COMPRESSION JOB PROGRESS")


class TranspilationJobModel(BaseJobModel):
    """Job model for transpilation jobs."""

    circuit_ids: list
    device_id: str
    transpilation_options: dict

    # Populated after job finishes, list of circuit IDs
    transpiled_circuit_ids: Optional[list] = None

    def retrieve_status(self) -> JobStatus:
        """
        Query backend for the job status.
        """
        response = cast(TranspilationJobModel, self._retrieve_status())
        self.transpiled_circuit_ids = response.transpiled_circuit_ids

        return self.status

    def result(self) -> list[CircuitModel] | None:
        """
        Return job result - CircuitModel.
        Block and wait for the job to complete.
        """
        super().result()

        return [self._client.get_circuit(circuit_id=c_id) for c_id in self.transpiled_circuit_ids]

    def progress(self):
        """
        Display the progress widget, stream logs from the job.
        """
        self._progress("TRANSPILATION JOB PROGRESS")


class SubmitTranspilationModel(BaseModel):
    """Data model for transpilation submit request"""

    experiment_id: str
    circuit_ids: list = []
    device_id: str
    transpilation_options: Optional[dict] = {}
    name: Optional[str] = ""
    description: Optional[str] = ""


class SubmitObservableBackpropagationModel(BaseModel):
    """Data model for observable backpropagation submit request"""

    circuit_id: str
    observables: List[List[Tuple[str, float, float]]]
    max_qwc_groups: Optional[int] = None
    max_error_total: Optional[float] = None
    max_error_per_slice: Optional[float] = None


class ObservableBackpropagationModel(BaseModel):
    """Data model for observable backpropagation response"""

    optimized_circuit_ids: List[str]
    backpropagated_observables: List[List[Tuple[str, float, float]]]


class DeviceModel(BaseModel, ParseIterableMixin):
    """Haiqu supported devices"""

    id: str
    vendor: str
    name: str
    qubits: int
    status: str
    simulator: bool
    last_updated: Optional[datetime] = None
    pending_jobs: Optional[int] = None
    operation_names: Optional[List[str]] = None
    coupling_map: Optional[List[List[int]]] = None


class PostprocessParams(BaseModel):
    """Parameters for postprocessing optimization."""

    postprocess_iterations: int = Field(default=5, ge=1, description="Number of optimization passes (must be >= 1)")
    use_fast_eval: bool = Field(default=True, description="Enable fast evaluation")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed (must be >= 0 if provided)")


class PostprocessRequest(BaseModel):
    """Request model for postprocessing endpoint."""

    lp_problem: str  # QUBO serialized as LP file content
    counts: dict[str, Union[int, float]]
    params: Optional[dict] = None  # Parameters as dict for API compatibility


class PostprocessResponse(BaseModel):
    """Response model for postprocessing endpoint."""

    optimized_costs: dict[str, float]
    optimized_counts: dict[str, Union[int, float]]


class PostprocessSKQDParams(BaseModel):
    """Parameters for SKQD postprocessing (SQD diagonalization)."""

    samples_per_batch: int = Field(default=100, ge=1, description="Samples per SQD batch")
    num_batches: int = Field(default=5, ge=1, description="Number of subsampling batches")
    max_iterations: int = Field(default=15, ge=1, description="Max SQD self-consistent iterations")
    symmetrize_spin: bool = Field(default=True, description="Enforce spin symmetry in SQD")
    configuration_recovery: bool = Field(default=False, description="Apply configuration recovery to refine noisy bitstrings")
    seed: Optional[int] = Field(default=None, ge=0, description="Random seed")


class PostprocessSKQDRequest(BaseModel):
    """Request model for SKQD postprocessing endpoint."""

    h1e: list[list[float]]
    h2e: list[list[list[list[float]]]]
    results: list[dict[str, float]]
    num_shots: int
    norb: int
    nelec: Tuple[int, int]
    params: Optional[dict] = None


class PostprocessSKQDResponse(BaseModel):
    """Response model for SKQD postprocessing endpoint."""

    energy: float
    subspace_dimension: int
    iteration_history: list[dict] = Field(
        default=[],
        description=(
            "Per-iteration results. Each dict has: "
            "'energy' (float, best energy across batches), "
            "'subspace_dimension' (int, subspace dim of best batch), "
            "'runtime' (float, seconds for the full iteration)."
        ),
    )


class SKQDResult:
    """Result of an SKQD postprocess call.

    Attributes:
        energy: Best ground-state energy estimate across all iterations.
        subspace_dimension: Dimension of the CI subspace for the best energy.
        amplitudes: CI coefficients (1-D array). The ground state is
            reconstructed as
            |psi> = sum_i amplitudes[i] |ci_strs_a[i]> |ci_strs_b[i]>.
        ci_strs_a: Alpha-spin determinant strings (1-D integer array).
            Each integer encodes which orbitals are occupied by alpha
            electrons. To read orbital j, check ``(integer >> j) & 1``.
        ci_strs_b: Beta-spin determinant strings (1-D integer array).
            Same encoding as ci_strs_a but for beta electrons.
        orbital_occupancies_alpha: Expected alpha occupation per orbital
            (1-D array of length norb).
        orbital_occupancies_beta: Expected beta occupation per orbital
            (1-D array of length norb).
        iteration_history: List of per-iteration dicts, each with keys
            'energy' (float), 'subspace_dimension' (int), 'runtime' (float).

    Example:
        For norb=3 the determinant integer encodes occupations as::

            integer  binary   occupied orbitals
            -------  ------   -----------------
            5        101      orbitals 0 and 2
            3        011      orbitals 0 and 1
            6        110      orbitals 1 and 2

        A 3-orbital Hubbard model with nelec=(2, 2) might return::

            result = SKQDResult(
                energy=-4.862,
                subspace_dimension=9,
                amplitudes=[0.872, -0.345, -0.345, 0.012, ...],
                ci_strs_a=[3, 5, 6, 3, ...],  # 3 -> orbs 0,1 | 5 -> orbs 0,2 | 6 -> orbs 1,2
                ci_strs_b=[3, 5, 6, 5, ...],
                orbital_occupancies_alpha=[0.78, 0.66, 0.56],
                orbital_occupancies_beta=[0.78, 0.66, 0.56],
                iteration_history=[
                    {"energy": -4.750, "subspace_dimension": 9, "runtime": 0.02},
                    {"energy": -4.862, "subspace_dimension": 9, "runtime": 0.01},
                ],
            )

        To convert a determinant integer to a list of occupied orbitals::

            det = 5  # norb=3
            occupied = [j for j in range(norb) if (det >> j) & 1]
            # occupied == [0, 2]
    """

    def __init__(
        self,
        energy: float,
        subspace_dimension: int,
        iteration_history: Optional[list[dict]] = None,
        amplitudes: Optional[list] = None,
        ci_strs_a: Optional[list] = None,
        ci_strs_b: Optional[list] = None,
        orbital_occupancies_alpha: Optional[list] = None,
        orbital_occupancies_beta: Optional[list] = None,
    ):
        self.energy = energy
        self.subspace_dimension = subspace_dimension
        self.iteration_history = iteration_history or []
        self.amplitudes = np.asarray(amplitudes).ravel() if amplitudes is not None else None
        self.ci_strs_a = np.asarray(ci_strs_a, dtype=np.int64).ravel() if ci_strs_a is not None else None
        self.ci_strs_b = np.asarray(ci_strs_b, dtype=np.int64).ravel() if ci_strs_b is not None else None
        self.orbital_occupancies_alpha = np.array(orbital_occupancies_alpha) if orbital_occupancies_alpha is not None else None
        self.orbital_occupancies_beta = np.array(orbital_occupancies_beta) if orbital_occupancies_beta is not None else None

    def best_energy_per_iteration(self) -> list[float]:
        """Return the best (minimum) energy from each iteration.

        Useful for plotting convergence across SQD iterations.
        """
        return [it["energy"] for it in self.iteration_history]

    def best_subspace_dimension_per_iteration(self) -> list[int]:
        """Return the subspace dimension corresponding to the best energy per iteration."""
        return [it["subspace_dimension"] for it in self.iteration_history]

    def runtimes_per_iteration(self) -> list[float]:
        """Return the runtime in seconds for each iteration."""
        return [it["runtime"] for it in self.iteration_history]

    def save_to_json(self, path: str) -> None:
        """Save result to a JSON file."""
        data = {
            "energy": self.energy,
            "subspace_dimension": self.subspace_dimension,
            "iteration_history": self.iteration_history,
        }
        if self.amplitudes is not None:
            data["amplitudes"] = self.amplitudes.tolist()
        if self.ci_strs_a is not None:
            data["ci_strs_a"] = self.ci_strs_a.tolist()
        if self.ci_strs_b is not None:
            data["ci_strs_b"] = self.ci_strs_b.tolist()
        if self.orbital_occupancies_alpha is not None:
            data["orbital_occupancies_alpha"] = self.orbital_occupancies_alpha.tolist()
        if self.orbital_occupancies_beta is not None:
            data["orbital_occupancies_beta"] = self.orbital_occupancies_beta.tolist()
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_json(cls, path: str) -> "SKQDResult":
        """Load result from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            energy=data["energy"],
            subspace_dimension=data["subspace_dimension"],
            iteration_history=data.get("iteration_history", []),
            amplitudes=data.get("amplitudes"),
            ci_strs_a=data.get("ci_strs_a"),
            ci_strs_b=data.get("ci_strs_b"),
            orbital_occupancies_alpha=data.get("orbital_occupancies_alpha"),
            orbital_occupancies_beta=data.get("orbital_occupancies_beta"),
        )

    def __repr__(self):
        parts = [
            f"  energy={self.energy}",
            f"  subspace_dimension={self.subspace_dimension}",
            f"  num_iterations={len(self.iteration_history)}",
        ]
        if self.amplitudes is not None:
            parts.append(f"  num_determinants={len(self.amplitudes)}")
        if self.orbital_occupancies_alpha is not None:
            parts.append(
                f"  orbital_occupancies_alpha={np.array2string(self.orbital_occupancies_alpha, precision=4, separator=', ')}"
            )
        if self.orbital_occupancies_beta is not None:
            parts.append(
                f"  orbital_occupancies_beta={np.array2string(self.orbital_occupancies_beta, precision=4, separator=', ')}"
            )
        history = self.iteration_history
        if history:
            if len(history) <= 4:
                lines = [f"    {it}" for it in history]
            else:
                lines = [f"    {it}" for it in history[:2]]
                lines.append(f"    ... ({len(history) - 4} more iterations)")
                lines.extend(f"    {it}" for it in history[-2:])
            parts.append("  iteration_history=[\n" + ",\n".join(lines) + "\n  ]")
        return "SKQDResult(\n" + ",\n".join(parts) + "\n)"


class SKQDSubmitModel(BaseModel):
    """Data model for SKQD postprocessing job submission."""

    experiment_id: str
    name: str = "SKQD Postprocessing"
    h1e: list[list[float]]
    h2e: list[list[list[list[float]]]]
    results: list[dict[str, float]]
    num_shots: int
    norb: int
    nelec: Tuple[int, int]
    params: Optional[dict] = None


class SKQDJobModel(BaseJobModel):
    """Job model for SKQD postprocessing jobs.

    Output data (energy, subspace_dimension, iteration_history) is stored
    in the inherited ``info`` dict so that adding or removing output fields
    does not require a database migration.
    """

    # Input data (stored in DB for the worker to pick up)
    h1e: Optional[list[list[float]]] = None
    h2e: Optional[list[list[list[list[float]]]]] = None
    results: Optional[list[dict[str, float]]] = None
    num_shots: Optional[int] = None
    norb: Optional[int] = None
    nelec: Optional[Tuple[int, int]] = None
    params: Optional[dict] = None

    # Output data lives in self.info (populated by worker when done)

    def retrieve_status(self) -> JobStatus:
        """Query backend for the job status."""
        self._retrieve_status()
        return self.status

    def result(self) -> Union[SKQDResult, None]:
        """Return the SKQD postprocessing result.

        Blocks until the job completes.

        Returns:
            SKQDResult: Object with energy, CI state, orbital occupancies, and
                iteration history. See :class:`SKQDResult` for field details.

        Raises:
            Exception: If the job failed or was cancelled.

        Job's ``info`` field contains the output data:
            * energy - best ground-state energy estimate
            * subspace_dimension - CI subspace dimension for the best energy
            * amplitudes - CI coefficients of the ground state
            * ci_strs_a / ci_strs_b - alpha/beta determinant strings
            * orbital_occupancies_alpha / orbital_occupancies_beta - per-orbital occupancies
            * iteration_history - per-iteration dicts with energy, subspace_dimension, runtime
        """
        super().result()

        info = self.info or {}
        return SKQDResult(
            energy=info.get("energy"),
            subspace_dimension=info.get("subspace_dimension"),
            iteration_history=info.get("iteration_history", []),
            amplitudes=info.get("amplitudes"),
            ci_strs_a=info.get("ci_strs_a"),
            ci_strs_b=info.get("ci_strs_b"),
            orbital_occupancies_alpha=info.get("orbital_occupancies_alpha"),
            orbital_occupancies_beta=info.get("orbital_occupancies_beta"),
        )

    def progress(self):
        """Display the progress widget, stream logs from the job."""
        self._progress("SKQD POSTPROCESSING JOB PROGRESS")


class LRQAOACircuitSubmitModel(BaseModel):
    """Submit model for LR-QAOA circuit generation."""

    experiment_id: str
    lp_problem: str  # QUBO serialized as LP file content
    p: int = Field(ge=1, description="Number of QAOA layers (must be >= 1)")
    initial_state_qpy: Optional[str] = None  # QPY dump of initial state circuit
    alphas: Optional[list[float]] = Field(default=None, description="Cost operator parameters (length must match p)")
    betas: Optional[list[float]] = Field(default=None, description="Mixer operator parameters (length must match p)")
    delta: float = Field(default=0.5, description="Ramp parameter (typical values are between 0 and 1)")
    name: str

    def model_post_init(self, __context):
        """Validate alphas and betas lengths match p, and warn if delta is unusual."""
        super().model_post_init(__context)
        if self.alphas is not None and len(self.alphas) != self.p:
            raise ValueError(f"alphas length ({len(self.alphas)}) must match p ({self.p})")
        if self.betas is not None and len(self.betas) != self.p:
            raise ValueError(f"betas length ({len(self.betas)}) must match p ({self.p})")


class VariationalProblemSubmitModel(BaseModel):
    """Submit model for variational problem execution."""

    experiment_id: str
    circuit_id: str  # The logged ansatz circuit ID
    observable: Tuple[List[str], List[float]]  # ([pauli_strings], [coefficients])
    shots: int = 1000
    device_id: str
    options: Optional[dict] = None
    initial_parameters: Optional[List[float]] = None
    optimizer_options: OptimizerOptionsUnion = Field(default_factory=NFTOptimizerOptions)
    use_mitigation: bool = False
    use_compression: bool = False
    compression_options: Optional[dict] = None
    name: Optional[str] = ""
    description: Optional[str] = ""


class VariationalCompressionStats:
    """Per-step compression statistics from a variational optimization run."""

    def __init__(
        self,
        per_step_quality: List[float],
        per_step_percent: Optional[List[float]] = None,
        **_ignored,
    ):
        self.per_step_quality = per_step_quality
        self.per_step_percent = per_step_percent or []
        self.mean_quality = float(sum(per_step_quality) / len(per_step_quality)) if per_step_quality else 0.0
        self.mean_compression_percent = (
            float(sum(self.per_step_percent) / len(self.per_step_percent)) if self.per_step_percent else 0.0
        )

    def __repr__(self):
        return (
            f"VariationalCompressionStats("
            f"mean_quality={self.mean_quality:.4f}, "
            f"mean_compression_percent={self.mean_compression_percent:.1f}%, "
            f"steps={len(self.per_step_quality)})"
        )


class VariationalResult:
    """Result of a variational optimization job.

    Attributes:
        optimal_parameters: The optimized parameter values.
        min_loss: The minimum loss value found.
        loss_history: Loss values recorded during optimization. Cadence depends
            on the optimizer: NFT records one entry per optimizer iteration
            (parameter update); scipy methods record one entry per objective /
            circuit evaluation (so length can exceed ``maxiter`` and is capped
            by ``maxfev``).
        weights_history: Weight vectors parallel to ``loss_history`` —
            ``weights_history[i]`` is the weight vector associated with
            ``loss_history[i]``.
        compression_stats: Optional per-step compression statistics when
            compression-in-training was enabled.
    """

    def __init__(
        self,
        *,
        optimal_parameters: List[float],
        min_loss: float,
        loss_history: List[float],
        weights_history: Optional[List[List[float]]] = None,
        compression_stats: Optional[VariationalCompressionStats] = None,
    ):
        self.optimal_parameters = optimal_parameters
        self.min_loss = min_loss
        self.loss_history = loss_history
        self.weights_history = weights_history
        self.compression_stats = compression_stats

    @staticmethod
    def _format_vector(values: List[float], *, max_vals: int = 5) -> str:
        if len(values) > max_vals:
            preview = ", ".join(f"{v:.6g}" for v in values[:max_vals])
            return f"[{preview}, ...] ({len(values)} values)"
        return repr(values)

    @staticmethod
    def _format_history(values: List[float], *, max_entries: int = 5) -> str:
        if len(values) > max_entries:
            preview = [f"{v:.6f}" for v in values[:max_entries]]
            return f"[{', '.join(preview)}, ...] ({len(values)} entries)"
        return f"[{', '.join(f'{v:.6f}' for v in values)}]"

    @classmethod
    def _format_weights_history(cls, weights_history: Optional[List[List[float]]], *, max_entries: int = 5) -> str:
        if weights_history is None:
            return "None"
        if len(weights_history) > max_entries:
            return f"[...] ({len(weights_history)} entries)"
        return "[" + ", ".join(cls._format_vector(w) for w in weights_history) + "]"

    def __repr__(self):
        return (
            f"VariationalResult(\n"
            f"  min_loss={self.min_loss},\n"
            f"  optimal_parameters={self._format_vector(self.optimal_parameters)},\n"
            f"  loss_history={self._format_history(self.loss_history)},\n"
            f"  weights_history={self._format_weights_history(self.weights_history)}\n"
            f")"
        )


class VariationalJobModel(BaseJobModel):
    """Job returned for variational problem execution."""

    circuit_id: str
    observable: Tuple[List[str], List[float]]  # ([pauli_strings], [coefficients])
    shots: int
    device_id: str
    options: Optional[dict] = None
    optimizer_options: Optional[dict] = None
    use_mitigation: bool = False

    # Populated after job finishes
    optimal_parameters: Optional[List[float]] = None
    min_loss: Optional[float] = None
    loss_history: Optional[List[float]] = None
    weights_history: Optional[List[List[float]]] = None
    compression_stats: Optional[dict] = None

    def retrieve_status(self) -> JobStatus:
        """Query backend for the job status."""
        response = cast(VariationalJobModel, self._retrieve_status())
        self.optimal_parameters = response.optimal_parameters
        self.min_loss = response.min_loss
        self.loss_history = response.loss_history
        self.weights_history = response.weights_history
        self.compression_stats = response.compression_stats
        return self.status

    def result(self) -> Union[VariationalResult, None]:
        """Return the optimization result.

        Blocks until the job completes.

        Returns:
            VariationalResult | None: The optimization result, or ``None`` if not
            available (e.g. from a dry run job). See ``VariationalResult`` for
            attribute details.

        Raises:
            Exception: If the job failed or was cancelled.

        Job's ``info`` field contains additional information (not every field may be present):

            * loss_history - history of losses during the optimization (NFT: per
              iteration; scipy: per circuit evaluation)
            * weights_history - weight vectors parallel to loss_history
        """
        super().result()

        if self.optimal_parameters is None or self.min_loss is None or self.loss_history is None:
            return None

        comp_stats = None
        if self.compression_stats:
            comp_stats = VariationalCompressionStats(**self.compression_stats)

        return VariationalResult(
            optimal_parameters=self.optimal_parameters,
            min_loss=self.min_loss,
            loss_history=self.loss_history,
            weights_history=self.weights_history,
            compression_stats=comp_stats,
        )

    @property
    def qpu_cost(self) -> Optional[dict]:
        """
        Actual QPU cost after execution. Pricing varies by vendor
        (e.g. time-based for IBM, shot-based for AWS).
        Returns None if there is no QPU cost (e.g. simulator execution).

        Returns:
            dict or None: ``{"native": {"amount": <float>, "unit": "<vendor_unit>"},
            "converted": {"amount": <dollars>, "unit": "USD"}}``

        Warning:
            This property blocks the current CPU thread until the job reaches a
            terminal state (Done/Error), similarly to ``result()``. Use ``retrieve_status()``
            to first check the state of the job to omit the prolonged thread blockade.
        """
        self.result()

        if self.info is None:
            return None
        return self.info.get("qpu_cost")

    @property
    def session_cost(self) -> Optional[dict]:
        """
        Total session cost after execution, based on IBM Quantum Runtime Session
        wall-clock time. See `IBM Session documentation
        <https://docs.quantum.ibm.com/run/sessions>`_ for details.
        Returns None if there is no session cost (e.g. simulator execution).

        Returns:
            dict or None: ``{"native": {"amount": <seconds>, "unit": "s"},
            "converted": {"amount": <dollars>, "unit": "USD"}}``

        Warning:
            This property blocks the current CPU thread until the job reaches a
            terminal state (Done/Error), similarly to ``result()``. Use ``retrieve_status()``
            to first check the state of the job to omit the prolonged thread blockade.
        """
        self.result()

        if self.info is None:
            return None
        return self.info.get("session_cost")

    @property
    def estimated_qpu_cost(self) -> Optional[dict]:
        """
        Estimated QPU cost for the variational optimization, computed from the
        ansatz circuit, shots, optimizer's maximum iteration count, and the device
        cost model. Populated when the job was submitted with ``dry_run=True``;
        otherwise ``None``.

        Returns:
            dict or None: ``{"native": {"amount": <seconds|shots>, "unit": "<vendor_unit>"},
            "converted": {"amount": <dollars>, "unit": "USD"},
            "warning": <str | None>}``. When the job was submitted with
            ``use_session=True``, ``warning`` is populated with the string
            ``"Session mode also bills classical optimization and parameter-update
            time, not included here."``

        Warning:
            This property blocks the current CPU thread until the job reaches a
            terminal state (Done/Error), similarly to ``result()``. Use ``retrieve_status()``
            to first check the state of the job to omit the prolonged thread blockade.
        """
        self.result()

        if self.info is None:
            return None
        return self.info.get("estimated_qpu_cost")

    def progress(self):
        """Display the progress widget, stream logs from the job."""
        self._progress("VARIATIONAL JOB PROGRESS")


class PretrainingJobType(enum.Enum):
    """Class for `pretrain_type`."""

    PRETRAIN = "Pretrain"
    MPS_GRADIENT = "MpsGradient"


class PretrainingSubmitModel(BaseModel):
    """Submit model for pretraining variational circuits."""

    experiment_id: str
    circuit_id: str  # The logged ansatz circuit ID
    loss_expression: str  # sympy objective as a string; "x" for a linear single-observable problem
    observables: Dict[str, Tuple[List[str], List[float]]]  # {symbol_name: ([term_strings], [coefficients])}
    name: Optional[str] = ""
    description: Optional[str] = ""
    max_time: Optional[float] = None
    seed: Optional[int] = None
    initial_parameters: Optional[List[float]] = None
    options: Optional[dict] = None
    pretrain_type: str = PretrainingJobType.PRETRAIN.value


class PretrainingJobModel(BaseJobModel):
    """Job model for pretraining variational circuits."""

    circuit_id: str
    loss_expression: str  # sympy objective as a string; "x" for a linear single-observable problem
    observables: Dict[str, Tuple[List[str], List[float]]]  # {symbol_name: ([term_strings], [coefficients])}
    max_time: Optional[float] = None
    seed: Optional[int] = None
    initial_parameters: Optional[List[float]] = None
    options: Optional[dict] = None
    pretrain_type: PretrainingJobType = PretrainingJobType.PRETRAIN

    # Populated after job finishes
    result_vector: Optional[List[float]] = None  # optimized weights for pretrain, or gradient vector for gradient
    loss: Optional[float] = None  # observable expectation value, only populated for gradient

    def retrieve_status(self) -> JobStatus:
        """Query backend for the job status."""
        response = cast(PretrainingJobModel, self._retrieve_status())
        self.result_vector = response.result_vector
        self.loss = response.loss
        return self.status

    def result(self) -> List | Tuple[float, List[float]] | None:
        """Return the pretraining result.

        Blocks until the job completes.

        Returns:
            For ``pretrain`` jobs: ``List[float] | None`` — optimized ansatz weights, or ``None`` if
            interrupted.
            For ``gradient`` jobs: ``Tuple[float, List[float]] | None`` — ``(loss, gradient)``
            where ``loss`` is the observable expectation value and ``gradient`` is the list of partial
            derivatives (one per ansatz parameter), or ``None`` if interrupted.

        Raises:
            Exception: If the job failed or was cancelled.
        """
        super().result()

        if self.pretrain_type == PretrainingJobType.PRETRAIN:
            return self.result_vector
        elif self.pretrain_type == PretrainingJobType.MPS_GRADIENT:
            if self.result_vector is None:
                # matches keyboard interruption behavior where no result is yet retrieved
                # and job.result() supposed to return just None
                return None
            return self.loss, self.result_vector
        else:
            raise ValueError(f"Unrecognized pretrain type: {self.pretrain_type}")

    def progress(self):
        """Display the progress widget, stream logs from the job."""
        if self.pretrain_type == PretrainingJobType.MPS_GRADIENT:
            pretrain_type_words = "GRADIENT"
        else:
            pretrain_type_words = "".join([(" " + s if s.isupper() else s) for s in self.pretrain_type.value]).strip()
            pretrain_type_words = pretrain_type_words.upper()
        self._progress(f"{pretrain_type_words} JOB PROGRESS")


class QECReferenceModel(BaseModel):
    authors: list[str]
    title: str
    year: int
    doi: str | None = None
    annotation: str | None = None


class QECCodeModel(BaseModel):
    """Pydantic model for the QEC Code Library API responses."""

    slug: str
    name: str
    family: str
    sub_family: str = ""
    code_type: str
    n: int | None = None
    k: int | None = None
    d: int | dict | None = None
    completeness: str
    description: str | None = None
    stabilizers: dict | None = None
    logical_operators: dict | None = None
    layout_svg_url: str | None = None
    qecx_url: str | None = None
    references: list[QECReferenceModel] | None = None

    class Config:
        from_attributes = True


class QECCodeListResponseModel(BaseModel):
    """Pydantic model for the QEC Code Library list API response."""

    items: list[QECCodeModel]
    total: int
    page: int
    page_size: int


JOB_MODELS = (
    AnalyticsJobModel
    | BaseJobModel
    | DataLoadingJobModel
    | HybridJobModel
    | LocalJobModel
    | RunJobModel
    | StateCompressionJobModel
    | TranspilationJobModel
    | VariationalJobModel
    | PretrainingJobModel
    | SKQDJobModel
)
