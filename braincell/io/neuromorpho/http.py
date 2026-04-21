# Copyright 2026 BrainX Ecosystem Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Tiny retry/backoff helper around a ``requests`` session.

Used by :class:`NeuroMorphoClient` for every JSON API call. Streaming
file downloads issue a single request without retries because resuming
partial downloads is out of scope for this module.
"""



import random
import time
from typing import Any, Callable

from .errors import NeuroMorphoHTTPError, NeuroMorphoNotFoundError

__all__ = [
    "MAX_BACKOFF_SECONDS",
    "request_with_retry",
]

#: Upper bound on a single backoff sleep, in seconds.
MAX_BACKOFF_SECONDS = 30.0


def _is_transient_status(status: int) -> bool:
    return status >= 500


def _classify_exception(exc: BaseException) -> str:
    """Return ``"transient"``, ``"permanent"``, or ``"unknown"`` for *exc*."""

    name = type(exc).__name__
    module = type(exc).__module__
    if module.startswith("requests"):
        if name in {"ConnectionError", "Timeout", "ReadTimeout", "ConnectTimeout"}:
            return "transient"
        if name in {"HTTPError", "TooManyRedirects", "URLRequired", "MissingSchema"}:
            return "permanent"
    return "unknown"


def _sleep_for_attempt(attempt: int, backoff_base: float, sleep: Callable[[float], None]) -> None:
    delay = backoff_base * (2 ** attempt)
    delay = min(delay, MAX_BACKOFF_SECONDS)
    delay += random.uniform(0.0, backoff_base)
    sleep(delay)


def request_with_retry(
    session: Any,
    url: str,
    *,
    timeout: float,
    attempts: int = 3,
    backoff_base: float = 0.5,
    params: dict[str, Any] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Issue a GET request with retry on transient failures.

    Retries on connection / timeout errors and on HTTP responses with
    status >= 500. A 404 is translated to
    :class:`NeuroMorphoNotFoundError` immediately. Other 4xx responses
    raise :class:`NeuroMorphoHTTPError` without retrying. After
    ``attempts`` failures the most recent error is wrapped in
    :class:`NeuroMorphoHTTPError`.

    Parameters
    ----------
    session : requests.Session-like
        Object exposing ``get(url, params=..., timeout=...)``. Tests
        pass a fake.
    url : str
        Absolute URL to GET.
    timeout : float
        Per-attempt timeout passed to ``session.get``.
    attempts : int
        Total number of attempts (including the first). Must be ``>= 1``.
    backoff_base : float
        Base for exponential backoff. Sleep before retry ``n`` is
        ``backoff_base * 2 ** n`` seconds plus jitter, capped at
        :data:`MAX_BACKOFF_SECONDS`.
    params : dict or None
        Query string parameters.
    sleep : callable
        Injection point for tests; defaults to :func:`time.sleep`.

    Returns
    -------
    requests.Response-like
        The successful HTTP response (status < 400).

    Raises
    ------
    NeuroMorphoNotFoundError
        On HTTP 404.
    NeuroMorphoHTTPError
        On non-2xx response after exhausted retries, or on a permanent
        4xx error.
    """

    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts!r}")

    last_exc: BaseException | None = None
    last_status: int | None = None
    for attempt in range(attempts):
        try:
            response = session.get(url, params=params, timeout=timeout)
        except Exception as exc:  # noqa: BLE001
            kind = _classify_exception(exc)
            last_exc = exc
            if kind == "permanent" or attempt == attempts - 1:
                raise NeuroMorphoHTTPError(
                    f"GET {url} failed: {exc}",
                    status=0,
                    url=url,
                ) from exc
            _sleep_for_attempt(attempt, backoff_base, sleep)
            continue

        status = int(getattr(response, "status_code", 0) or 0)
        if status == 404:
            raise NeuroMorphoNotFoundError(
                f"NeuroMorpho.Org returned 404 for {url}",
                url=url,
            )
        if status and status < 400:
            return response
        if status and not _is_transient_status(status):
            raise NeuroMorphoHTTPError(
                f"GET {url} returned HTTP {status}",
                status=status,
                url=url,
            )
        last_status = status
        if attempt == attempts - 1:
            raise NeuroMorphoHTTPError(
                f"GET {url} returned HTTP {status} after {attempts} attempts",
                status=status,
                url=url,
            )
        _sleep_for_attempt(attempt, backoff_base, sleep)

    # Defensive: the loop above always returns or raises.
    raise NeuroMorphoHTTPError(
        f"GET {url} failed after {attempts} attempts (last_status={last_status}, last_exc={last_exc})",
        status=last_status or 0,
        url=url,
    )
