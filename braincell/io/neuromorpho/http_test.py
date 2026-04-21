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

"""Tests for :mod:`braincell.io.neuromorpho.http`."""

import unittest

from braincell.io.neuromorpho import (
    NeuroMorphoHTTPError,
    NeuroMorphoNotFoundError,
)
from braincell.io.neuromorpho._testing import FakeResponse, FakeSession
from braincell.io.neuromorpho.http import request_with_retry


class RequestWithRetryTest(unittest.TestCase):
    def test_retries_on_transient_then_succeeds(self) -> None:
        session = FakeSession([
            FakeResponse(status_code=503),
            FakeResponse(status_code=503),
            FakeResponse(json_data={"ok": True}, status_code=200),
        ])
        sleeps: list[float] = []
        response = request_with_retry(
            session,
            "https://example.test/x",
            timeout=5.0,
            attempts=3,
            backoff_base=0.01,
            sleep=lambda d: sleeps.append(d),
        )
        self.assertEqual(response.json(), {"ok": True})
        self.assertEqual(len(session.calls), 3)
        self.assertEqual(len(sleeps), 2)

    def test_gives_up_after_attempts_exhausted(self) -> None:
        session = FakeSession([
            FakeResponse(status_code=503),
            FakeResponse(status_code=503),
        ])
        with self.assertRaises(NeuroMorphoHTTPError) as ctx:
            request_with_retry(
                session,
                "https://example.test/x",
                timeout=5.0,
                attempts=2,
                backoff_base=0.01,
                sleep=lambda d: None,
            )
        self.assertEqual(ctx.exception.status, 503)

    def test_404_raises_not_found_immediately(self) -> None:
        session = FakeSession([FakeResponse(status_code=404)])
        with self.assertRaises(NeuroMorphoNotFoundError):
            request_with_retry(
                session,
                "https://example.test/x",
                timeout=5.0,
                attempts=3,
                backoff_base=0.01,
                sleep=lambda d: None,
            )
        self.assertEqual(len(session.calls), 1)

    def test_4xx_other_than_404_does_not_retry(self) -> None:
        session = FakeSession([FakeResponse(status_code=400)])
        with self.assertRaises(NeuroMorphoHTTPError) as ctx:
            request_with_retry(
                session,
                "https://example.test/x",
                timeout=5.0,
                attempts=3,
                backoff_base=0.01,
                sleep=lambda d: None,
            )
        self.assertEqual(ctx.exception.status, 400)
        self.assertEqual(len(session.calls), 1)

    def test_attempts_must_be_positive(self) -> None:
        session = FakeSession([])
        with self.assertRaises(ValueError):
            request_with_retry(
                session,
                "https://example.test/x",
                timeout=5.0,
                attempts=0,
                backoff_base=0.01,
            )


if __name__ == "__main__":
    unittest.main()
