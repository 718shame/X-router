import logging
import os
import threading
import time
from itertools import cycle
from typing import Iterable, Optional

from dotenv import load_dotenv


logger = logging.getLogger(__name__)


class APIKeyPool:
    """Thread-safe API Key poller, supporting pooling and dynamic circuit breaking."""

    def __init__(self, keys: Optional[Iterable[str]] = None) -> None:
        load_dotenv()
        env_keys = os.getenv("API_BASE_KEYS")
        pool = list(keys or [])
        if env_keys:
            pool.extend(k.strip() for k in env_keys.split(",") if k.strip())
        if not pool:
            raise ValueError(
                "API_BASE API Key not detected, please configure in environment variable API_BASE_KEYS or pass explicitly."
            )

        self._lock = threading.Lock()
        self._keys = pool
        self._cycler = cycle(self._keys)

        # Consecutive failure threshold and cooldown time (seconds), can be overridden via environment variables
        self._fail_threshold = int(os.getenv("KEY_FAIL_THRESHOLD", "2"))
        self._cooldown_seconds = int(os.getenv("KEY_COOLDOWN_SEC", "300"))
        self._fail_counts = {key: 0 for key in self._keys}
        self._disabled_until = {key: 0.0 for key in self._keys}

    def _mark_disabled(self, key: str) -> None:
        until = time.time() + self._cooldown_seconds
        self._disabled_until[key] = until
        logger.warning("API Key %s consecutive failures, entering cooldown until %s", key[:12] + "...", time.ctime(until))

    def get_key(self) -> str:
        with self._lock:
            now = time.time()
            for _ in range(len(self._keys)):
                key = next(self._cycler)
                disabled_until = self._disabled_until.get(key, 0.0)
                if disabled_until and now < disabled_until:
                    continue
                return key
        raise RuntimeError("No available API_BASE API Key, please try again later or restore Key pool.")

    def mark_success(self, key: str) -> None:
        with self._lock:
            if key in self._fail_counts:
                self._fail_counts[key] = 0
                self._disabled_until[key] = 0.0

    def mark_failure(self, key: str) -> None:
        with self._lock:
            if key not in self._fail_counts:
                return
            self._fail_counts[key] += 1
            if self._fail_counts[key] >= self._fail_threshold:
                self._fail_counts[key] = 0
                self._mark_disabled(key)

    def add_key(self, key: str) -> None:
        if not key:
            return
        with self._lock:
            if key not in self._keys:
                self._keys.append(key)
                self._cycler = cycle(self._keys)
                self._fail_counts[key] = 0
                self._disabled_until[key] = 0.0

    def remove_key(self, key: str) -> None:
        with self._lock:
            if key in self._keys:
                self._keys.remove(key)
                self._fail_counts.pop(key, None)
                self._disabled_until.pop(key, None)
                if not self._keys:
                    raise RuntimeError("API Key pool is empty, unable to continue calling.")
                self._cycler = cycle(self._keys)