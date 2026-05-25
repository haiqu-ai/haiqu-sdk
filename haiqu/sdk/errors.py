"""
Haiqu SDK: error classes.
"""

import functools

from .utils import is_jupyter

from .exceptions import (
    NETWORK_EXCEPTIONS,
    APIKeyRequiredError,
    CircuitAnalyticsComputationError,
    CircuitNotFoundError,
    ExperimentSearchByNameError,
    InvalidAPIKeyError,
    InvalidFiltersError,
    LogError,
)

JUPYTER_LAB = is_jupyter()

DISPLAY_AS_HTML_EXCEPTIONS = (
    APIKeyRequiredError,
    CircuitAnalyticsComputationError,
    CircuitNotFoundError,
    ExperimentSearchByNameError,
    InvalidAPIKeyError,
    InvalidFiltersError,
)


def format_network_error(rest_api_uri: str, exception: Exception):
    if hasattr(exception, "response") and exception.response is not None and exception.response.status_code == 402:
        return (
            "Your account does not have enough credits to run jobs. "
            "Please request more credits http://dashboard.haiqu.ai/pricing"
        )

    return (
        f"Haiqu API service {rest_api_uri} is not responding or returned an error.\n"
        "Please check the technical details:\n\n"
        f"{exception}"
    )


def error_widget_or_string(msg: str) -> str | None:
    """
    Helper function to display error message as a widget in Jupyter Lab or
    return string in script mode.

    Args:
        msg (str): The error message to display or return.
    Returns:
        str | None: The error message string in script mode, or None if displayed as a widget in Jupyter Lab.
    """
    if JUPYTER_LAB:
        from haiqu.sdk.wiz.jupyter import graceful_error_widget

        graceful_error_widget(msg)
        return None
    return msg


def graceful_api_errors_message(func):
    """User-friendly API connection errors handler. Prints human-readable output.

    In Jupyter Lab it displays an error message, in the script mode - raises an exception.
    """

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # TODO: if target object has `_client` then retrieve `rest_api_uri`
        rest_api_uri = "api.haiqu.ai"

        try:
            if hasattr(self, "_check_login"):
                self._check_login()
            return func(self, *args, **kwargs)
        except NETWORK_EXCEPTIONS as e:
            if JUPYTER_LAB:
                error_widget_or_string(format_network_error(rest_api_uri, e))
                return
            raise e
        except DISPLAY_AS_HTML_EXCEPTIONS as e:
            if JUPYTER_LAB:
                error_widget_or_string(str(e))
                return
            raise e
        except LogError as e:
            if JUPYTER_LAB:
                from haiqu.sdk.wiz.jupyter import log_error_widget

                log_error_widget(str(e))
                return
            raise e
        except TypeError as e:
            if func.__name__ == "log" and JUPYTER_LAB:
                from haiqu.sdk.wiz.jupyter import log_error_widget

                log_error_widget(str(e))
                return
            raise e

    return wrapper
