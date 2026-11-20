"""
Haiqu SDK. Exceptions and errors.
"""

import functools

from requests.exceptions import ConnectionError, RequestException
from urllib3.exceptions import HTTPError
from urllib.error import URLError
import socket

NETWORK_EXCEPTIONS = (
    ConnectionError,
    HTTPError,
    RequestException,
    URLError,
    socket.timeout,
)


class APIKeyRequiredError(Exception):
    """
    Raised when API key is not provided in login() or in the OS environment.
    """


class InvalidFiltersError(Exception):
    """
    Raised when haiqu function got invalid filtering options.
    """


class InvalidAPIKeyError(Exception):
    """
    Raised when the communication with the REST API couldn't be made
    because of an invalid API key.
    """


class OutdatedSDKError(Exception):
    """Raised when the API requires a newer SDK version."""


class ExperimentSearchByNameError(Exception):
    """
    Raised when the request experiment by name failed with 404 error.
    """


class CircuitNotFoundError(Exception):
    """
    Raised when the Circuit-involving request failed with 404 error.
    """


class CircuitNotRegisteredInExperimentError(Exception):
    """
    Raised when the QuantumCircuit wasn't registered in the experiment.
    User should use `haiqu.log(<experiment name>, circuit)` to associate
    the circuit with the experiment."
    """


class CircuitAnalyticsComputationError(Exception):
    """
    Raised when the analytics computation job failed.
    """


class JobNotRegisteredInExperimentError(Exception):
    """
    Raised when the Qiskit job results wasn't registered in the experiment.
    User should use `haiqu.log(circuit, results)` to associate
    the results with the circuit."
    """


class LogError(Exception):
    """
    Raised when logging to the Haiqu cloud environment fails.
    """


class _JobEndedError(Exception):
    """A job that ended without a result: its ID and logs travel with the exception, through pickling and copying."""

    def __init__(self, message: str, *, job_id: str, logs: str | None = None):
        super().__init__(message)
        self.job_id = job_id
        self.logs = logs

    def __reduce__(self):
        # The default rebuilds the exception from its message alone, which the required job_id refuses.
        return functools.partial(type(self), job_id=self.job_id, logs=self.logs), self.args


class JobFailedError(_JobEndedError):
    """
    Raised by a job's ``result`` when the job failed.

    Attributes:
        job_id (str): ID of the job.
        logs (str | None): All log messages of the job, of which the exception message shows the error.
    """


class JobCancelledError(_JobEndedError):
    """
    Raised by a job's ``result`` when the job was cancelled, by the user or because it exceeded its time limit.

    Attributes:
        job_id (str): ID of the job.
        logs (str | None): All log messages of the job, of which the exception message shows the reason.
    """
