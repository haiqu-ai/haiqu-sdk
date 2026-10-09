"""
Haiqu SDK. Connection diagnostics reported by ``haiqu.health_check()``.
"""

import enum
import os
import platform
import socket
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

from requests import exceptions as requests_exceptions

from .api_client import ApiClient
from .exceptions import InvalidAPIKeyError, OutdatedSDKError
from .version import get_version

HEALTH_CHECK_TIMEOUT = 10  # seconds

# Environment variables that change how requests reach the network. Only their names are reported.
NETWORK_ENV_VARS = (
    "HTTPS_PROXY",
    "HTTP_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
    "SSL_CERT_FILE",
)


class CheckStatus(str, enum.Enum):
    """Outcome of a single health check."""

    OK = "OK"
    WARNING = "WARN"
    FAILED = "FAIL"
    SKIPPED = "SKIP"


@dataclass
class CheckResult:
    """Outcome of a single health check with a human-readable explanation."""

    name: str
    status: CheckStatus
    message: str


@dataclass
class HealthCheckContext:
    """Inputs shared by all health checks."""

    edge_uri: str
    api_access_key: str | None


@dataclass(frozen=True)
class HealthCheck:
    """A named check that runs only after the checks it requires have passed.

    A check returns its status and message. An exception raised by a check is reported as a failure of that check.
    """

    name: str
    run: Callable[[HealthCheckContext], tuple[CheckStatus, str]]
    requires: tuple[str, ...] = ()


@dataclass
class HealthReport:
    """Results of all health checks against one Haiqu API service."""

    edge_uri: str
    checks: list[CheckResult]

    def __str__(self) -> str:
        lines = [f"Haiqu health check: {self.edge_uri}"]
        lines += [f"  {'[' + check.status.value + ']':<8}{check.name:<16}{check.message}" for check in self.checks]
        return "\n".join(lines)


def _check_environment(ctx: HealthCheckContext) -> tuple[CheckStatus, str]:
    message = f"haiqu-sdk {get_version()}, Python {platform.python_version()}, {platform.platform(terse=True)}"
    network_vars = [name for name in NETWORK_ENV_VARS if os.environ.get(name) or os.environ.get(name.lower())]
    if network_vars:
        message += f"; network settings: {', '.join(network_vars)}"
    return CheckStatus.OK, message


def _check_api_reachable(ctx: HealthCheckContext) -> tuple[CheckStatus, str]:
    client = ApiClient(rest_api_uri=ctx.edge_uri, retry_total=0)
    start = time.perf_counter()
    version = client.get_api_version(timeout=HEALTH_CHECK_TIMEOUT)
    elapsed_ms = round((time.perf_counter() - start) * 1000)
    return CheckStatus.OK, f"Haiqu API {version} responded in {elapsed_ms} ms"


def _check_authentication(ctx: HealthCheckContext) -> tuple[CheckStatus, str]:
    if not ctx.api_access_key:
        return CheckStatus.SKIPPED, "Pass api_access_key to also check your API key or set the HAIQU_API_KEY environment variable"

    client = ApiClient(api_access_key=ctx.api_access_key, rest_api_uri=ctx.edge_uri, retry_total=0)
    try:
        username = client.health_auth(timeout=HEALTH_CHECK_TIMEOUT)
    except InvalidAPIKeyError:
        return CheckStatus.FAILED, "The API key is not valid"
    except OutdatedSDKError as e:
        return CheckStatus.FAILED, str(e)
    except requests_exceptions.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:
            return CheckStatus.WARNING, "This Haiqu API service cannot check API keys yet"
        raise
    return CheckStatus.OK, f"API key accepted for {username}"


HEALTH_CHECKS = (
    HealthCheck("environment", _check_environment),
    HealthCheck("api_reachable", _check_api_reachable),
    HealthCheck("authentication", _check_authentication, requires=("api_reachable",)),
)


def run_health_check(api_access_key: str | None = None, edge_uri: str | None = None) -> HealthReport:
    """Run all health checks and collect their results.

    Args:
        api_access_key (str | None): Haiqu API key to check.
        edge_uri (str | None): Haiqu API service to check.

    Returns:
        HealthReport: Result of every check, in the order they ran.
    """
    ctx = HealthCheckContext(edge_uri=edge_uri, api_access_key=api_access_key)
    results: dict[str, CheckResult] = {}

    for check in HEALTH_CHECKS:
        not_passed = [name for name in check.requires if results[name].status not in (CheckStatus.OK, CheckStatus.WARNING)]
        if not_passed:
            status, message = CheckStatus.SKIPPED, f"Skipped because {', '.join(not_passed)} did not pass"
        else:
            try:
                status, message = check.run(ctx)
            except Exception as e:
                status, message = CheckStatus.FAILED, _describe_error(e, edge_uri)
        results[check.name] = CheckResult(check.name, status, message)

    return HealthReport(edge_uri=edge_uri, checks=list(results.values()))


def _describe_error(error: Exception, edge_uri: str) -> str:
    """Explain a failed request in terms of what the user can check."""
    host = urlparse(edge_uri).hostname or edge_uri

    if isinstance(error, (requests_exceptions.JSONDecodeError, KeyError)):
        return f"{edge_uri} did not respond like the Haiqu API"
    if isinstance(error, (requests_exceptions.MissingSchema, requests_exceptions.InvalidSchema, requests_exceptions.InvalidURL)):
        return f"{edge_uri} is not a valid address"
    if isinstance(error, requests_exceptions.SSLError):
        return (
            f"Secure connection to {host} failed. If you are behind a corporate proxy, "
            "set REQUESTS_CA_BUNDLE to its certificate bundle"
        )
    if isinstance(error, requests_exceptions.ProxyError):
        return f"Could not connect to {host} through the configured proxy"
    if isinstance(error, requests_exceptions.Timeout):
        return f"No response from {host} within {HEALTH_CHECK_TIMEOUT} s"
    if isinstance(error, requests_exceptions.ConnectionError):
        if _is_name_resolution_error(error):
            return f"Could not resolve {host}. Check your internet connection and DNS settings"
        return f"Could not connect to {host}. Check your internet connection and firewall settings"
    if isinstance(error, requests_exceptions.HTTPError) and error.response is not None:
        status_code = error.response.status_code
        if status_code >= 500:
            return f"The Haiqu API is temporarily unavailable (HTTP {status_code})"
        return f"{host} responded with HTTP {status_code}"
    return f"{type(error).__name__}: {error}"


def _is_name_resolution_error(error: requests_exceptions.ConnectionError) -> bool:
    reason = getattr(error.args[0], "reason", None) if error.args else None
    cause = getattr(reason, "__cause__", None) or getattr(reason, "__context__", None)
    return isinstance(cause, socket.gaierror)
