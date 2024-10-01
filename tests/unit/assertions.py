"""Common assertions."""

from typing import Any, Callable


def assert_raises(
    call: Callable[[], Any], expected_error: type[BaseException], message: str
):
    """Assert that the provided call raises an expected error.

    Args:
        call: the function call to test
        expected_error: the expected error the call will raise
        message: the message to provide if the expected exception is not raised
    """
    raised_expected_error = False
    try:
        call()
    except expected_error:
        raised_expected_error = True
    assert raised_expected_error, message
