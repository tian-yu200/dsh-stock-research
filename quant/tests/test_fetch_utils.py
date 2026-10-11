# -*- coding: utf-8 -*-
"""Tests for ``fetch_utils.install_request_timeout`` (T9).

Root cause of the 2026-10-11 T2 hang: AkShare's eastmoney/tencent channels call
``requests.get(..., timeout=None)`` explicitly. ``kwargs.setdefault("timeout",
...)`` is a no-op when the key exists with value ``None``, so those requests
never timed out and one blocked socket froze the whole fetch wave. The wrapper
must override an explicit ``None`` but leave a real numeric timeout alone.

Every network call is bounded independently of the code under test: the request
runs in a daemon thread and is joined with a hard timeout, so a regression shows
up as a fast failure ("request blocked: timeout not applied"), never a hang.
"""
from __future__ import annotations

import pathlib
import socket
import sys
import threading
import time
import unittest

import requests

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_utils import install_request_timeout

_ORIGINAL_REQUEST = requests.sessions.Session.request

class _NeverReplyServer:
    """Accepts connections and never sends a byte (black-hole server)."""

    def __init__(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._sock.settimeout(0.2)
        while not self._stop.is_set():
            try:
                conn, _ = self._sock.accept()
            except (socket.timeout, OSError):
                continue
            # Hold the connection open and never reply.
            try:
                conn.settimeout(0.2)
                while not self._stop.is_set():
                    try:
                        if conn.recv(1):
                            pass
                    except (socket.timeout, OSError):
                        pass
            finally:
                try:
                    conn.close()
                except OSError:
                    pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._thread.join(timeout=3)

def _call_bounded(func, *, limit=10.0):
    """Run ``func`` in a daemon thread; return (result, exc, finished).

    If it does not finish within ``limit`` seconds we return finished=False so
    the caller can fail fast instead of hanging forever.
    """
    box: dict = {}

    def _target():
        try:
            box["result"] = func()
        except BaseException as exc:  # noqa: BLE001 - captured for the caller
            box["exc"] = exc

    thread = threading.Thread(target=_target, daemon=True)
    thread.start()
    thread.join(timeout=limit)
    return box.get("result"), box.get("exc"), not thread.is_alive()

class RequestTimeoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _NeverReplyServer()
        self.addCleanup(self.server.close)
        requests.sessions.Session.request = _ORIGINAL_REQUEST

    def tearDown(self) -> None:
        requests.sessions.Session.request = _ORIGINAL_REQUEST

    def _url(self) -> str:
        return "http://127.0.0.1:%d/slow" % self.server.port

    def test_explicit_none_timeout_is_overridden(self) -> None:
        # RED before the fix: the request blocks forever (finished=False).
        install_request_timeout(2.0)
        start = time.monotonic()
        _, exc, finished = _call_bounded(
            lambda: requests.get(self._url(), timeout=None), limit=10.0)
        elapsed = time.monotonic() - start
        self.assertTrue(finished, "request blocked: timeout not applied")
        self.assertIsInstance(exc, requests.exceptions.ReadTimeout)
        self.assertLess(elapsed, 3.5)

    def test_explicit_numeric_timeout_not_overridden(self) -> None:
        install_request_timeout(30.0)
        seen: dict = {}

        def spy(self, method, url, **kwargs):
            seen.update(kwargs)
            raise requests.exceptions.ReadTimeout("stop")

        requests.sessions.Session.request = spy
        _, exc, finished = _call_bounded(
            lambda: requests.get(self._url(), timeout=1.0), limit=10.0)
        self.assertTrue(finished)
        self.assertIsInstance(exc, requests.exceptions.ReadTimeout)
        self.assertEqual(seen.get("timeout"), 1.0)

    def test_wrapper_is_idempotent(self) -> None:
        requests.sessions.Session.request = _ORIGINAL_REQUEST
        install_request_timeout(2.0)
        first = requests.sessions.Session.request
        install_request_timeout(2.0)
        self.assertIs(requests.sessions.Session.request, first)


if __name__ == "__main__":
    unittest.main()
