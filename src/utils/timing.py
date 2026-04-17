import time
from contextlib import contextmanager
from typing import Iterator


@contextmanager
def track_time() -> Iterator[callable]:
    start = time.perf_counter()

    def _elapsed() -> float:
        return time.perf_counter() - start

    try:
        yield _elapsed
    finally:
        pass
