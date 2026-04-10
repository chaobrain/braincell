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

"""Shared test helpers for the NeuroMorpho.Org package.

This module is intentionally private (leading underscore) and does not
follow the ``*_test.py`` naming convention, so pytest does not discover
it as a test file. The siblings ``*_test.py`` modules import from here
to share the ``FakeResponse`` / ``FakeSession`` doubles and the sample
payload factory.
"""

from pathlib import Path
from typing import Any

__all__ = [
    "FIXTURE_SWC",
    "FakeResponse",
    "FakeSession",
    "sample_neuron_payload",
]


#: Path to the minimal SWC fixture used by tests that need to actually
#: parse a morphology. The fixture lives under ``develop_doc/morpho_files/``,
#: matching the convention used by every other reader test.
FIXTURE_SWC = (
    Path(__file__).resolve().parents[3]
    / "develop_doc"
    / "morpho_files"
    / "three_points_soma.swc"
)


class FakeResponse:
    """Minimal stand-in for ``requests.Response`` used by ``FakeSession``.

    Parameters
    ----------
    json_data : Any
        Object returned by :meth:`json`. ``None`` makes :meth:`json`
        raise, signalling that the test was not expected to call it.
    content : iterable of bytes or None
        Chunks yielded by :meth:`iter_content`. Used by streaming
        download tests.
    url : str
        Reflected back through :attr:`url`.
    status_code : int
        HTTP status code returned by :meth:`raise_for_status` and read
        by the retry helper.
    """

    def __init__(
        self,
        *,
        json_data: Any = None,
        content: Any = None,
        url: str = "https://example.test",
        status_code: int = 200,
    ) -> None:
        self._json_data = json_data
        self._content = content or []
        self.url = url
        self.status_code = status_code

    def json(self) -> Any:
        if self._json_data is None:
            raise AssertionError("json() was not expected for this response")
        return self._json_data

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def iter_content(self, chunk_size: int = 8192):
        del chunk_size
        for chunk in self._content:
            yield chunk

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class FakeSession:
    """Test double for ``requests.Session``.

    Yields successive responses from a fixed list. A response item may
    be either a :class:`FakeResponse` or an exception instance, in which
    case the exception is raised on the call.

    Parameters
    ----------
    responses : iterable
        List of :class:`FakeResponse` instances or exceptions.

    Attributes
    ----------
    calls : list of (str, dict)
        Recorded ``(url, kwargs)`` for every ``get(...)`` call.
    """

    def __init__(self, responses) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, **kwargs):
        self.calls.append((url, kwargs))
        if not self.responses:
            raise AssertionError(f"Unexpected GET {url!r}")
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def sample_neuron_payload(**overrides: Any) -> dict[str, Any]:
    """Return a representative NeuroMorpho.Org neuron payload.

    Used by every test that needs a :class:`NeuroMorphoNeuron`. Pass
    keyword arguments to override individual fields.

    Parameters
    ----------
    **overrides : Any
        Fields to replace on the base payload.

    Returns
    -------
    dict
    """

    payload: dict[str, Any] = {
        "neuron_id": 10047,
        "neuron_name": "TypeA-10",
        "archive": "Scanziani",
        "species": "mouse",
        "brain_region": ["neocortex", "occipital", "layer 6"],
        "cell_type": ["principal cell"],
        "original_format": "TypeA-10.asc",
        "png_url": "http://neuromorpho.org/images/typea-10.png",
        "_links": {
            "measurements": {"href": "http://neuromorpho.org/api/morphometry/id/10047"},
        },
    }
    payload.update(overrides)
    return payload
