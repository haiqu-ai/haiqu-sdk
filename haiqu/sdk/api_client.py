"""
Haiqu SDK.
Client for Haiqu REST API service.
"""

import functools
import types
from collections.abc import Mapping
from typing import Optional, List, Union, Dict, Tuple, TYPE_CHECKING
from urllib.parse import urljoin

import requests
from requests import codes, exceptions
from requests.adapters import HTTPAdapter
from json import dumps
from urllib3.util import Retry

from . import schemas
from .constants import REST_API_URI, SDK_VERSION_HEADER
from .exceptions import InvalidAPIKeyError, CircuitNotFoundError, OutdatedSDKError
from .utils import HaiquJSONEncoder
from .version import get_version

if TYPE_CHECKING:
    from .optimization import QUBO

API_ACTIONS = {
    "user": "/user",
    "benchmarks": "/help/benchmarks",
    "list_experiments": "/experiments",
    "init_experiment": "/experiments/init",
    "create_experiment": "/experiments",
    "get_experiment": "/experiments/{experiment_id}",
    "update_experiment": "/experiments/{experiment_id}",
    "submit_experiment_metrics": "/experiments/{experiment_id}/metrics",
    "submit_circuit": "/circuits",
    "backpropagate_observables": "/circuits/{circuit_id}/backpropagate_observables",
    "list_circuits": "/experiments/{experiment_id}/circuits",
    "get_circuit": "/circuits/{circuit_id}",
    "compute_analytics": "/circuits/{circuit_id}/compute_analytics/{analytics_type}",
    "get_circuit_by_hash": "/experiments/{experiment_id}/circuits/{circuit_hash}",
    "get_metrics": "/circuits/{circuit_id}/metrics",
    "transpile_circuit": "/circuits/transpile",
    "build_lr_qaoa_circuit": "/circuits/lr_qaoa",
    "submit_circuit_metrics": "/experiments/{experiment_id}/circuits/{circuit_hash}/metrics",
    "get_metrics_evolution": "/circuits/{circuit_id}/metrics_evolution",
    "list_jobs": "/experiments/{experiment_id}/jobs",
    "submit_job": "/jobs",
    "submit_job_metrics": "/experiments/{experiment_id}/jobs/{job_id}/metrics",
    "get_job": "/jobs/{job_id}",
    "cancel_job": "/jobs/{job_id}/cancel",
    "restart_job": "/jobs/{job_id}/restart",
    "get_local_job_by_hash": "/experiments/{experiment_id}/jobs/{job_hash}",
    "list_devices": "/devices",
    "get_device": "/devices/{device_id}",
    "device_gate_map": "/devices/{device_id}/gate_map",
    "data_loading": "/data_loading",
    "data_loading_estimates": "/data_loading_estimates",
    "flow": "/flow",
    "run": "/run",
    "dry_run": "/dry-run",
    "hemistich": "/hemistich",
    "hemistich_estimates": "/hemistich_estimates",
    "postprocess": "/postprocess",
    "postprocess_skqd": "/postprocess_skqd",
    "variational_optimization": "/variational",
    "pretraining": "/pretraining",
}


def auto_set_current_client(cls):
    """
    Class decorator for ``ApiClient`` that sets ``_current_client`` for the duration of each method call.
    """

    def make_wrapper(attr):
        @functools.wraps(attr)
        def wrapper(self, *args, **kwargs):
            token = schemas._current_client.set(self)  # TODO (Python 3.14): Use set as context manager
            try:
                return attr(self, *args, **kwargs)
            finally:
                schemas._current_client.reset(token)

        return wrapper

    for name, attr in vars(cls).items():
        if not isinstance(attr, types.FunctionType):
            continue  # Skip everything that's not a regular method (including classmethod and staticmethod objects)
        setattr(cls, name, make_wrapper(attr))

    return cls


@auto_set_current_client
class ApiClient:
    """Haiqu REST API client class."""

    def __init__(
        self,
        api_access_key: str = "",
        rest_api_uri: str = REST_API_URI,
        retry_total: int = 3,
    ):
        """Haiqu API client init.

        Args:
            api_access_key (str): Haiqu API key, HAIQU_API_KEY.
            rest_api_uri (str): Base URI to REST API service.
            retry_total (int, optional): Total number of API request retries to allow.
        """
        self.api_access_key = api_access_key
        self.rest_api_uri = rest_api_uri
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        self.session.headers["authorization"] = api_access_key
        # Send the installed SDK version with every request.
        self.session.headers[SDK_VERSION_HEADER] = get_version()
        retries = Retry(
            total=retry_total,
            backoff_factor=0.5,
            allowed_methods={"GET", "POST", "PUT"},
            status_forcelist={502, 503, 504},
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))

    def get_user(self) -> schemas.UserModel:
        """Get Haiqu user's profile, API key perform identification.

        Returns:
            UserModel: User profile.
        """
        response = self._get(endpoint=API_ACTIONS["user"])
        return schemas.UserModel.model_validate_json(response.text)

    def init_experiment(self, data: schemas.ExperimentSubmitModel) -> schemas.ExperimentModel:
        """Handles experiment initialization."""
        response = self._post(
            endpoint=API_ACTIONS["init_experiment"],
            json=data.model_dump(),
        )

        return schemas.ExperimentModel.model_validate_json(response.text)

    def create_experiment(self, data: schemas.ExperimentSubmitModel) -> schemas.ExperimentModel:
        """Handles new experiment creating operation."""
        response = self._post(
            endpoint=API_ACTIONS["create_experiment"],
            json=data.model_dump(),
        )

        return schemas.ExperimentModel.model_validate_json(response.text)

    def update_experiment(self, experiment_id: str, data: schemas.ExperimentUpdateModel) -> schemas.ExperimentModel:
        """Handles experiment update operation."""
        response = self._put(
            endpoint=API_ACTIONS["update_experiment"].format(experiment_id=experiment_id),
            json=data.model_dump(),
        )

        return schemas.ExperimentModel.model_validate_json(response.text)

    def get_experiment(self, experiment_id: str = "", name: str = "") -> schemas.ExperimentModel:
        """Query/get the experiment from API by ID of Name."""
        if experiment_id:
            endpoint = API_ACTIONS["get_experiment"]
            response = self._get(endpoint=endpoint.format(experiment_id=experiment_id))
            return schemas.ExperimentModel.model_validate_json(response.text)
        elif name:
            response = self._get(endpoint=API_ACTIONS["list_experiments"], query_params={"name": name})
            experiments = schemas.ExperimentModel.parse_items(response.json())
            if len(experiments):
                return experiments[0]
            raise exceptions.HTTPError(f"404 Experiment {name} not found in Haiqu quantum service.")
        raise ValueError("Experiment ID or name required.")

    def submit_circuit(self, data: schemas.CircuitSubmitModel) -> schemas.CircuitModel:
        """Handles new circuit submit operation."""
        response = self._post(
            endpoint=API_ACTIONS["submit_circuit"],
            json=data.model_dump(),
        )
        circuit = schemas.CircuitModel.model_validate_json(response.text)
        return circuit

    def compute_analytics(
        self,
        circuit_id: str,
        analytics_type: schemas.AnalyticsType = schemas.AnalyticsType.CORE_METRICS,
    ):
        """Compute analytics for the circuit.

        Args:
            circuit_id (str): The database ID of the circuit record. Unique in the DB.
            analytics_type (schemas.AnalyticsType): The type of metrics to compute.

        Returns:
        """
        self._post(
            endpoint=API_ACTIONS["compute_analytics"].format(
                circuit_id=circuit_id,
                analytics_type=analytics_type.value,
            )
        )

    def submit_experiment_metrics(
        self,
        experiment_id: str,
        data: schemas.SubmitMetricsModel,
    ):
        """Submit experiment metrics."""
        self._post(
            endpoint=API_ACTIONS["submit_experiment_metrics"].format(experiment_id=experiment_id),
            json=data.model_dump(),
        )

    def submit_circuit_metrics(
        self,
        experiment_id: str,
        circuit_hash: str,
        data: schemas.SubmitMetricsModel,
    ):
        """Submit circuit metrics."""
        self._post(
            endpoint=API_ACTIONS["submit_circuit_metrics"].format(
                experiment_id=experiment_id,
                circuit_hash=circuit_hash,
            ),
            json=data.model_dump(),
        )

    def submit_job_metrics(
        self,
        experiment_id: str,
        job_hash: str,
        data: schemas.SubmitMetricsModel,
    ):
        """Submit job metrics."""
        self._post(
            endpoint=API_ACTIONS["submit_job_metrics"].format(
                experiment_id=experiment_id,
                job_id=job_hash,
            ),
            json=data.model_dump(),
        )

    def backpropagate_observables(
        self,
        submit_data: schemas.SubmitObservableBackpropagationModel,
    ) -> schemas.CircuitModel:
        """Backpropagate observables through the given circuit and log the new circuit to the current experiment.

        Args:
            submit_data (SubmitObservableBackpropagationModel): The data for observable backpropagation.
        Returns:
            CircuitModel: The new quantum circuit with backpropagated observables, logged to the current experiment.
        """
        response = self._post(
            endpoint=API_ACTIONS["backpropagate_observables"].format(circuit_id=submit_data.circuit_id),
            json=submit_data.model_dump(),
        )

        return schemas.ObservableBackpropagationModel.model_validate_json(response.text)

    def transpile_circuit(
        self,
        submit_data: schemas.SubmitTranspilationModel,
    ) -> schemas.TranspilationJobModel:
        """Transpile the given circuit to target device and log the transpiled circuit to the current experiment.

        Args:
            circuit (CircuitModel): The quantum circuit to transpile.
            device (DeviceModel | str): The target device or device ID to transpile the circuit to.
            **kwargs: Additional parameters for logging the circuit if not yet logged.

        Returns:
            TranspilationJobModel: The transpilation job.
        """
        response = self._post(endpoint=API_ACTIONS["transpile_circuit"], json=submit_data.model_dump())
        job = schemas.TranspilationJobModel.model_validate_json(response.text)
        return job

    def get_circuit(self, experiment_id: str = "", circuit_id: str = "", circuit_hash: str = "") -> schemas.CircuitModel:
        """Get circuit from API service by circuit ID or HASH.

        Args:
            experiment_id (str): The database ID of the parent experiment. Unique in the DB.
            circuit_id (str): The database ID of the circuit record. Unique in the DB.
            circuit_hash (str): The HASH of the circuit. Different experiments may have circuits with the same HASH.

        Returns:
            CircuitModel: The circuit data.
        """
        if not (circuit_id or circuit_hash):
            raise ValueError("Circuit ID or HASH required.")
        if circuit_id:
            endpoint = API_ACTIONS["get_circuit"].format(circuit_id=circuit_id)
        else:
            endpoint = API_ACTIONS["get_circuit_by_hash"].format(
                experiment_id=experiment_id,
                circuit_hash=circuit_hash,
            )
        response = self._get(endpoint=endpoint)
        circuit = schemas.CircuitModel.model_validate_json(response.text)
        return circuit

    def submit_job(self, data: schemas.LocalJobSubmitModel) -> schemas.LocalJobModel:
        """Handles local job/results submit operation.

        Args:
            data (LocalJobSubmitModel): Data of the local job (or result object).

        Returns:
            LocalJobModel: New job ID and URL to the Haiqu Dashboard.
        """
        response = self._post(
            endpoint=API_ACTIONS["submit_job"],
            json=data.model_dump(),
        )

        return schemas.LocalJobModel.model_validate_json(response.text)

    def get_job(self, experiment_id: str = "", job_id: str = "", job_hash: str = "") -> schemas.BaseJobModel:
        """Get job from API service by job ID or HASH.

        Args:
            experiment_id (str): The database ID of the parent experiment. Unique in the DB.
            job_id (str): The database ID of the circuit record. Unique in the DB.
            job_hash (str): The Hash of the job (only applies to the local jobs).

        Returns:
            BaseJobModel: The job data.
        """
        if not (job_id or job_hash):
            raise ValueError("Job ID or HASH required.")
        if job_id:
            endpoint = API_ACTIONS["get_job"].format(job_id=job_id)
        else:
            endpoint = API_ACTIONS["get_local_job_by_hash"].format(
                experiment_id=experiment_id,
                job_hash=job_hash,
            )
        response = self._get(endpoint=endpoint)
        base_job = schemas.BaseJobModel.model_validate_json(response.text)
        job_class = self._match_job_type(base_job.job_type)
        return job_class.model_validate_json(response.text)

    def cancel_job(self, job_id: str) -> schemas.BaseJobModel:
        """Cancel the job by job ID.

        Args:
            job_id (str): The database ID of the job record. Unique in the DB.

        Returns:
            BaseJobModel: The updated job data.
        """
        endpoint = API_ACTIONS["cancel_job"].format(job_id=job_id)
        response = self._post(endpoint=endpoint)
        return schemas.BaseJobModel.model_validate_json(response.text)

    def restart_job(self, job_id: str) -> schemas.BaseJobModel:
        """Restart the job by job ID.

        Args:
            job_id (str): The database ID of the job record. Unique in the DB.

        Returns:
            BaseJobModel: The updated job data with `JobStatus.SUBMITTED` if the job was successfully restarted.
        """
        endpoint = API_ACTIONS["restart_job"].format(job_id=job_id)
        response = self._post(endpoint=endpoint)
        return schemas.BaseJobModel.model_validate_json(response.text)

    def get_device(self, device_id: str) -> schemas.DeviceModel:
        """Get device from API service by device ID.

        Args:
            device_id (str): The ID of the device.
        Returns:
            DeviceModel: The device data.
        """
        endpoint = API_ACTIONS["get_device"].format(device_id=device_id)
        response = self._get(endpoint=endpoint)
        return schemas.DeviceModel.model_validate_json(response.text)

    def list_devices(self) -> list[schemas.DeviceModel]:
        """Get devices list"""
        response = self._get(endpoint=API_ACTIONS["list_devices"])
        return schemas.DeviceModel.parse_items(response.json())

    def list_experiments(self) -> list[schemas.ExperimentModel]:
        """Get user's experiments list"""
        response = self._get(endpoint=API_ACTIONS["list_experiments"])
        return schemas.ExperimentModel.parse_items(response.json())

    def list_circuits(
        self,
        experiment_id: str,
        circuit_ids: list[str] | None,
        job_type: schemas.JobType | None,
        limit: int,
    ) -> list[schemas.CircuitModel]:
        """Get circuits list in the experiment.

        Args:
            experiment_id (str): Return the circuits for the provided experiment ID.
            circuit_ids (list[str] | None): If not ``None``, return only circuits with these IDs.
            job_type (JobType | None): Return circuits for the provided job type in the experiment.
            limit (int): Limit the number of the circuits returned.
        """
        job_type_value = None
        if job_type:
            job_type_value = job_type.value

        response = self._get(
            endpoint=API_ACTIONS["list_circuits"].format(experiment_id=experiment_id),
            query_params={
                "limit": limit,
                "circuit_ids": circuit_ids,
                "job_type": job_type_value,
            },
        )
        return schemas.CircuitModel.parse_items(response.json())

    def list_jobs(
        self,
        experiment_id: str,
        job_type: schemas.JobType = None,
        circuit: Optional[Union[schemas.CircuitModel, str]] = None,
        limit: int = 10,
    ) -> list:
        """Get jobs list in the experiment or the circuit.

        Args:
            experiment_id (str): The ID of the experiment.
            job_type (JobType): Filter jobs by specific type. Default: no filtering.
            circuit (CircuitModel | str | None): If not ``None``, only show jobs related to the given circuit.
            limit (int): Limit the number of jobs returned.
        """
        job_type_value = None
        if job_type:
            job_type_value = job_type.value

        circuit_id = circuit
        if isinstance(circuit, schemas.CircuitModel):
            circuit_id = circuit.id

        try:
            response = self._get(
                endpoint=API_ACTIONS["list_jobs"].format(experiment_id=experiment_id),
                query_params={
                    "job_type": job_type_value,
                    "limit": limit,
                    "circuit_id": circuit_id,
                },
            )
            if job_type:
                job_class = self._match_job_type(job_type)
                return job_class.parse_items(response.json())

            # Response with different job types
            jobs = []
            for job_data in response.json():
                job_type = schemas.JobType(job_data.get("job_type"))
                job_class = self._match_job_type(job_type)
                job = job_class.model_validate(job_data)
                jobs.append(job)
            return jobs
        except exceptions.HTTPError as e:
            if circuit_id and e.response.status_code == 404:
                raise CircuitNotFoundError(f"The Circuit with ID {circuit_id} is not found in Haiqu quantum service.")
            else:
                raise e

    def data_loading(self, data: schemas.DataLoadingSubmitModel) -> schemas.DataLoadingJobModel:
        """Post Data Loading request"""
        response = self._post(endpoint=API_ACTIONS["data_loading"], json=data.model_dump())
        job = schemas.DataLoadingJobModel.model_validate_json(response.text)
        return job

    def data_loading_estimates(self, data: schemas.DataLoadingSubmitModel) -> schemas.DataLoadingEstimatesModel:
        """Post Data Loading estimates request"""
        response = self._post(endpoint=API_ACTIONS["data_loading_estimates"], json=data.model_dump())
        return schemas.DataLoadingEstimatesModel.model_validate_json(response.text)

    def compression_estimates(self, data: schemas.StateCompressionEstimatesSubmitModel) -> schemas.StateCompressionEstimatesModel:
        """Post compression estimates request"""
        response = self._post(endpoint=API_ACTIONS["hemistich_estimates"], json=data.model_dump())
        return schemas.StateCompressionEstimatesModel.model_validate_json(response.text)

    def flow(self, data: schemas.HybridSubmitModel) -> schemas.HybridJobModel:
        """Post Flow (hybrid program) request"""
        response = self._post(endpoint=API_ACTIONS["flow"], json=data.model_dump())
        job = schemas.HybridJobModel.model_validate_json(response.text)
        return job

    def run(self, data: schemas.RunSubmitModel) -> schemas.RunJobModel:
        """Post Run request"""
        response = self._post(endpoint=API_ACTIONS["run"], json=data.model_dump())
        job = schemas.RunJobModel.model_validate_json(response.text)
        return job

    def dry_run(self, data: schemas.RunSubmitModel) -> schemas.JobInsights:
        """Post Dry Run request"""
        response = self._post(endpoint=API_ACTIONS["dry_run"], json=data.model_dump())
        payload = response.json()
        if isinstance(payload, dict) and "job" in payload:
            job_payload = payload["job"]
        else:
            # Backward compatibility for older API servers returning RunJobModel directly.
            # until we update the API server to return JobInsights.
            job_payload = payload
            payload = {}

        job = schemas.RunJobModel.model_validate(job_payload)
        insights_payload = {"job": job}
        if "metrics" in payload:
            insights_payload["metrics"] = payload["metrics"]
        if "data" in payload:
            insights_payload["data"] = payload["data"]
        return schemas.JobInsights.model_validate(insights_payload)

    def compression(self, data: schemas.StateCompressionSubmitModel) -> list[schemas.StateCompressionJobModel]:
        """Post compression request"""
        response = self._post(endpoint=API_ACTIONS["hemistich"], json=data.model_dump())
        return schemas.StateCompressionJobModel.parse_items(response.json())

    def pretraining(self, data: schemas.PretrainingSubmitModel) -> schemas.PretrainingJobModel:
        """Post pretraining job request."""
        response = self._post(endpoint=API_ACTIONS["pretraining"], json=data.model_dump())
        job = schemas.PretrainingJobModel.model_validate_json(response.text)
        return job

    def variational_optimization(self, data: schemas.VariationalProblemSubmitModel) -> schemas.VariationalJobModel:
        """Post variational optimization job request."""
        response = self._post(endpoint=API_ACTIONS["variational_optimization"], json=data.model_dump())
        job = schemas.VariationalJobModel.model_validate_json(response.text)
        return job

    def build_lr_qaoa_circuit(
        self,
        data: schemas.LRQAOACircuitSubmitModel,
    ) -> schemas.CircuitModel:
        """Build LR-QAOA circuit via API (synchronous).

        Args:
            data: LR-QAOA circuit generation parameters

        Returns:
            CircuitModel: The generated circuit (ready to use immediately)
        """
        response = self._post(endpoint=API_ACTIONS["build_lr_qaoa_circuit"], json=data.model_dump())
        circuit = schemas.CircuitModel.model_validate_json(response.text)
        return circuit

    def api_postprocess(
        self,
        counts: Dict[str, Union[int, float]],
        problem: "QUBO",
        postprocess_iterations: int = 5,
        use_fast_eval: bool = True,
        seed: Optional[int] = None,
    ) -> Tuple[Dict[str, float], Dict[str, Union[int, float]]]:
        """
        Apply post-processing using Haiqu API's optimized algorithms.

        Uses Haiqu's proprietary optimization algorithm on the server side.

        Args:
            counts: Dictionary mapping bitstrings to their measurement counts.
                   Bitstrings must be in Qiskit's little-endian convention (rightmost bit = qubit 0).
            problem: The QUBO problem instance.
            postprocess_iterations: Maximum number of optimization passes. Defaults to 5.
            use_fast_eval: Enable fast evaluation. Defaults to True.
            seed: Random seed for reproducible results. Defaults to None.

        Returns:
            Tuple of (optimized_costs, optimized_counts):
                - optimized_costs: Dict mapping bitstrings to their costs
                - optimized_counts: Dict mapping bitstrings to aggregated counts

        Note:
            Bitstrings use Qiskit convention: rightmost bit = qubit 0.

        Raises:
            HTTPException: If the API request fails.
        """

        # Create request payload
        # Note: params is a dict for API compatibility
        # Validation happens in quantum_haiqu.py using PostprocessParams
        request_data = schemas.PostprocessRequest(
            lp_problem=problem.to_lp_string(),
            counts=counts,
            params={
                "postprocess_iterations": postprocess_iterations,
                "use_fast_eval": use_fast_eval,
                "seed": seed,
            },
        )

        # Make API request
        response = self._post(endpoint=API_ACTIONS["postprocess"], json=request_data.model_dump())

        # Parse response
        result = schemas.PostprocessResponse.model_validate_json(response.text)
        return result.optimized_costs, result.optimized_counts

    def api_postprocess_skqd(
        self,
        data: schemas.SKQDSubmitModel,
    ) -> schemas.SKQDJobModel:
        """
        Submit an SKQD postprocessing job (async worker).

        Args:
            data: SKQDSubmitModel with Hamiltonian tensors, results, and SQD parameters.

        Returns:
            SKQDJobModel: The submitted job (poll with .result() or .progress()).
        """
        response = self._post(
            endpoint=API_ACTIONS["postprocess_skqd"],
            json=data.model_dump(),
        )
        job = schemas.SKQDJobModel.model_validate_json(response.text)
        return job

    # ------------------------------------------------------------------------
    # Helper functions
    def _build_url(self, endpoint: str) -> str:
        """Build full REST API endpoint URL."""
        return urljoin(self.rest_api_uri, endpoint)

    def _get(
        self,
        endpoint: str,
        query_params: Optional[Mapping] = None,
        timeout=None,
    ) -> requests.Response:
        """Helper method for GET requests.

        Args:
            endpoint (str): API endpoint.
            query_params (dict): Get request params.

        Returns:
            requests.Response: API response.
        """
        response = self.session.get(
            url=self._build_url(endpoint),
            params=query_params,
            timeout=timeout,
        )
        self._handle_http_errors(response)
        return response

    def _post(
        self,
        endpoint: str,
        json: Optional[Mapping] = None,
    ) -> requests.Response:
        """Helper method for POST requests.

        Args:
            endpoint (str): API endpoint.
            json (dict): Post request body.

        Returns:
            requests.Response: API response.
        """
        data = dumps(json, cls=HaiquJSONEncoder)
        response = self.session.post(
            url=self._build_url(endpoint),
            data=data,
        )
        self._handle_http_errors(response)
        return response

    def _put(
        self,
        endpoint: str,
        json: Optional[Mapping] = None,
    ) -> requests.Response:
        """Helper method for PUT requests.

        Args:
            endpoint (str): API endpoint.
            json (dict): Put request body.

        Returns:
            requests.Response: API response.
        """
        data = dumps(json, cls=HaiquJSONEncoder)
        response = self.session.put(
            url=self._build_url(endpoint),
            data=data,
        )
        self._handle_http_errors(response)
        return response

    def _delete(self, endpoint: str) -> requests.Response:
        """Helper method for DELETE requests.

        Args:
            endpoint (str): API endpoint.

        Returns:
            requests.Response: API response.

        """

        response = self.session.delete(
            url=self._build_url(endpoint),
        )
        self._handle_http_errors(response)

        return response

    def _handle_http_errors(self, response: requests.Response) -> None:
        """Raise SDK-specific exceptions for failed API responses.

        Args:
            response (requests.Response): API response to inspect.

        Raises:
            OutdatedSDKError: Raised for the API's ``outdated_sdk`` response.
            InvalidAPIKeyError: Raised for ``401`` responses.
            requests.HTTPError: Raised by ``response.raise_for_status()`` for
                other non-OK responses.
        """
        if response.status_code == codes.UPGRADE_REQUIRED:  # 426
            payload = response.json()
            detail = payload["detail"]
            if detail.get("error_code") == "outdated_sdk":
                raise OutdatedSDKError(detail["message"])

        if response.status_code == codes.UNAUTHORIZED:  # 401
            raise InvalidAPIKeyError()
        elif not response.ok:
            response.raise_for_status()

    @staticmethod
    def _match_job_type(job_type: schemas.JobType) -> type:
        """
        Helper for matching job type to the corresponding job model class.
        Used in get_job() and list_jobs() to parse the job with the correct model based on its type.
        """
        match job_type:
            case schemas.JobType.ANALYTICS:
                job_class = schemas.AnalyticsJobModel
            case schemas.JobType.HYBRID:
                job_class = schemas.HybridJobModel
            case schemas.JobType.RUN:
                job_class = schemas.RunJobModel
            case schemas.JobType.DATA_LOADING:
                job_class = schemas.DataLoadingJobModel
            case schemas.JobType.COMPRESSION:
                job_class = schemas.StateCompressionJobModel
            case schemas.JobType.TRANSPILATION:
                job_class = schemas.TranspilationJobModel
            case schemas.JobType.LOCAL:
                job_class = schemas.LocalJobModel
            case schemas.JobType.VARIATIONAL:
                job_class = schemas.VariationalJobModel
            case schemas.JobType.PRETRAINING:
                job_class = schemas.PretrainingJobModel
            case schemas.JobType.SKQD:
                job_class = schemas.SKQDJobModel
            case _:
                job_class = schemas.BaseJobModel
        return job_class

    def help_benchmarks(self) -> List[schemas.ReferenceModel]:
        """Returns reference benchmarks."""
        response = self._get(endpoint=API_ACTIONS["benchmarks"])
        return schemas.ReferenceModel.parse_items(response.json())
