from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Full, Queue
from typing import Generic, Iterator, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class _Value(Generic[T]):
    value: T


@dataclass(frozen=True)
class _Error:
    error: BaseException


class _Done:
    pass


class InferenceWorker:
    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="qwen-local-inference",
        )
        self._state = threading.local()

    def run(self, fn: Callable[[], T]) -> T:
        if getattr(self._state, "active", False):
            return fn()

        future = self._executor.submit(self._run, fn)
        return future.result()

    def iterate(self, fn: Callable[[], Iterator[T]]) -> Iterator[T]:
        if getattr(self._state, "active", False):
            yield from fn()
            return

        queue: Queue[_Value[T] | _Error | _Done] = Queue(maxsize=1)
        stop_event = threading.Event()
        future = self._executor.submit(self._iterate, fn, queue, stop_event)
        completed = False

        try:
            while True:
                item = queue.get()
                if isinstance(item, _Done):
                    completed = True
                    future.result()
                    return
                if isinstance(item, _Error):
                    completed = True
                    future.result()
                    raise item.error
                yield item.value
        finally:
            if not completed:
                stop_event.set()
                future.result()

    def _run(self, fn: Callable[[], T]) -> T:
        self._state.active = True
        try:
            return fn()
        finally:
            self._state.active = False

    def _iterate(
        self,
        fn: Callable[[], Iterator[T]],
        queue: Queue[_Value[T] | _Error | _Done],
        stop_event: threading.Event,
    ) -> None:
        self._state.active = True
        iterator: Iterator[T] | None = None
        try:
            iterator = iter(fn())
            for item in iterator:
                if stop_event.is_set():
                    break
                if not self._put(queue, _Value(item), stop_event):
                    break
        except BaseException as exc:
            if not stop_event.is_set():
                self._put(queue, _Error(exc), stop_event)
            return
        finally:
            if iterator is not None:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
            self._state.active = False

        if not stop_event.is_set():
            self._put(queue, _Done(), stop_event)

    @staticmethod
    def _put(
        queue: Queue[_Value[T] | _Error | _Done],
        item: _Value[T] | _Error | _Done,
        stop_event: threading.Event,
    ) -> bool:
        while not stop_event.is_set():
            try:
                queue.put(item, timeout=0.1)
                return True
            except Full:
                continue
        return False
