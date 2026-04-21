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

"""Exception hierarchy for the NeuroMorpho.Org client.

Defines a small typed hierarchy so callers can ``except NeuroMorphoError``
without importing ``requests.exceptions``.
"""



__all__ = [
    "NeuroMorphoError",
    "NeuroMorphoHTTPError",
    "NeuroMorphoNotFoundError",
]


class NeuroMorphoError(Exception):
    """Base class for every error raised by the NeuroMorpho.Org client."""


class NeuroMorphoHTTPError(NeuroMorphoError):
    """Raised when an HTTP request to NeuroMorpho.Org fails permanently.

    This exception is also used to signal exhausted retries on transient
    network failures (in which case ``status`` is ``0``).

    Parameters
    ----------
    message : str
        Human-readable error description.
    status : int
        HTTP status code, or ``0`` when the failure was a connection /
        timeout error rather than an HTTP response.
    url : str
        The URL that produced the error.

    Attributes
    ----------
    status : int
    url : str
    """

    def __init__(self, message: str, *, status: int, url: str) -> None:
        super().__init__(message)
        self.status = int(status)
        self.url = str(url)


class NeuroMorphoNotFoundError(NeuroMorphoHTTPError):
    """Raised when a NeuroMorpho.Org resource (typically a neuron id) does not exist.

    This corresponds to an HTTP 404 response on requests such as
    ``/api/neuron/id/<neuron_id>``.
    """

    def __init__(self, message: str, *, url: str) -> None:
        super().__init__(message, status=404, url=url)
