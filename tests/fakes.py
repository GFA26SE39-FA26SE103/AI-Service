from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


class FakeReader:
    def __init__(self, reads: Iterable[tuple[bool, np.ndarray | None]]) -> None:
        self._reads = deque(reads)
        self._last_success: tuple[bool, np.ndarray | None] | None = None
        self.closed = False
        self.close_calls = 0
        self.read_calls = 0
        self._lock = threading.Lock()

    def read(self) -> tuple[bool, np.ndarray | None]:
        with self._lock:
            self.read_calls += 1
            if self.closed:
                return False, None
            if self._reads:
                result = self._reads.popleft()
                if result[0]:
                    self._last_success = result
                return result
            if self._last_success is not None:
                return self._last_success
            return False, None

    def close(self) -> None:
        with self._lock:
            self.closed = True
            self.close_calls += 1


class FakeTracker:
    def __init__(self) -> None:
        self.annotated_values: list[int] = []

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        self.annotated_values.append(int(frame[0, 0, 0]))
        return frame

    def encode_jpeg(self, frame: np.ndarray) -> bytes:
        return f"jpeg:{int(frame[0, 0, 0])}".encode()


@dataclass
class RecordingFactories:
    reader_sequences: deque[list[tuple[bool, np.ndarray | None]]]
    readers: list[FakeReader] = field(default_factory=list)
    trackers: list[FakeTracker] = field(default_factory=list)
    supplied_passwords: list[str | None] = field(default_factory=list)

    @classmethod
    def with_sequences(
        cls, *sequences: list[tuple[bool, np.ndarray | None]]
    ) -> "RecordingFactories":
        return cls(deque(sequences))

    def reader_factory(self, request):
        self.supplied_passwords.append(request.password.get_secret_value() if request.password else None)
        sequence = self.reader_sequences.popleft() if self.reader_sequences else [(False, None)]
        reader = FakeReader(sequence)
        self.readers.append(reader)
        return reader

    def tracker_factory(self, _request):
        tracker = FakeTracker()
        self.trackers.append(tracker)
        return tracker


def frame(value: int) -> np.ndarray:
    return np.full((2, 2, 3), value, dtype=np.uint8)

