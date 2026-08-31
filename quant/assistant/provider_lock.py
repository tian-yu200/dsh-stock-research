"""Cross-process lock for an atomically replaced Qlib provider."""
from __future__ import annotations

from pathlib import Path

from filelock import FileLock, Timeout


class ProviderLock:
    """Coordinate provider readers with the process that publishes a rebuild.

    FileLock is deliberately used for both modes. A provider read is one
    exclusive lease, which keeps atomic directory replacement safe on Windows
    and avoids a racy hand-rolled reader/writer protocol.
    """

    def __init__(
        self,
        provider_dir: Path,
        *,
        purpose: str,
        exclusive: bool = False,
    ) -> None:
        target = Path(provider_dir).resolve()
        self.lock_path = target.parent / f".{target.name}.provider.lock"
        self.purpose = str(purpose)
        self.exclusive = bool(exclusive)
        self._lock = FileLock(str(self.lock_path))
        self._acquired = False

    def acquire(self, *, timeout_seconds: float = 30.0) -> "ProviderLock":
        try:
            self._lock.acquire(timeout=max(0.0, float(timeout_seconds)))
        except Timeout as exc:
            mode = "write" if self.exclusive else "read"
            raise TimeoutError(
                f"timed out acquiring Qlib provider {mode} lock for {self.purpose}"
            ) from exc
        self._acquired = True
        return self

    def release(self) -> None:
        if not self._acquired:
            return
        self._lock.release()
        self._acquired = False

    def __enter__(self) -> "ProviderLock":
        return self.acquire()

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()
