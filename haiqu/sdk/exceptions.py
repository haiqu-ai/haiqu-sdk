"""
Haiqu SDK. Exceptions and errors.
"""

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
