from __future__ import annotations

import gc
import logging
import re
import threading
import time
from collections.abc import Callable
from typing import Any

KeepAliveSeconds = float | None

_DURATION_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+))\s*([smhd]?)\s*$")
_DURATION_MULTIPLIERS = {
    "": 1.0,
    "s": 1.0,
    "m": 60.0,
    "h": 3600.0,
    "d": 86400.0,
}


class KeepAliveParseError(ValueError):
    pass


class RuntimeLifecycle:
    def __init__(
        self,
        *,
        name: str,
        is_loaded: Callable[[], bool],
        unload: Callable[[], None],
        inference_lock: Any,
        logger: logging.Logger,
    ) -> None:
        self._name = name
        self._is_loaded = is_loaded
        self._unload = unload
        self._inference_lock = inference_lock
        self._logger = logger
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._generation = 0
        self._unload_at: float | None = None
        self._last_keep_alive: KeepAliveSeconds = None

    def begin_request(self) -> None:
        with self._lock:
            self._generation += 1
            self._cancel_timer_locked()
            self._unload_at = None

    def finish_request(self, keep_alive: KeepAliveSeconds) -> None:
        unload_now = False

        with self._lock:
            self._generation += 1
            self._cancel_timer_locked()
            self._last_keep_alive = keep_alive

            if not self._is_loaded():
                self._unload_at = None
                return

            if keep_alive is None:
                self._unload_at = None
                self._logger.info("%s model will remain loaded", self._name)
                return

            if keep_alive <= 0:
                self._unload_at = None
                unload_now = True
            else:
                self._schedule_unload_locked(keep_alive, self._generation)

        if unload_now and self._is_loaded():
            self._logger.info("unloading %s model immediately", self._name)
            self._unload()

    def state(self) -> dict[str, Any]:
        with self._lock:
            unload_at = self._unload_at
            keep_alive = self._last_keep_alive

        unload_in = None
        if unload_at is not None:
            unload_in = max(0.0, unload_at - time.monotonic())

        return {
            "loaded": self._is_loaded(),
            "keep_alive_seconds": keep_alive,
            "unload_in_seconds": None if unload_in is None else round(unload_in, 3),
        }

    def _cancel_timer_locked(self) -> None:
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def _schedule_unload_locked(self, delay: float, generation: int) -> None:
        self._unload_at = time.monotonic() + delay
        timer = threading.Timer(delay, self._unload_if_idle, args=(generation,))
        timer.daemon = True
        self._timer = timer
        timer.start()
        self._logger.info("%s model scheduled to unload in %.2fs", self._name, delay)

    def _unload_if_idle(self, generation: int) -> None:
        if not self._inference_lock.acquire(blocking=False):
            self._reschedule_if_current(generation, delay=1.0)
            return

        try:
            with self._lock:
                if generation != self._generation:
                    return
                self._timer = None
                self._unload_at = None

            if self._is_loaded():
                self._logger.info("unloading idle %s model", self._name)
                self._unload()
        finally:
            self._inference_lock.release()

    def _reschedule_if_current(self, generation: int, *, delay: float) -> None:
        with self._lock:
            if generation != self._generation:
                return
            self._schedule_unload_locked(delay, generation)


def parse_keep_alive(value: Any, default: str) -> KeepAliveSeconds:
    effective = default if value is None else value

    if isinstance(effective, bool):
        raise KeepAliveParseError("keep_alive must be a duration string or seconds")

    if isinstance(effective, (int, float)):
        seconds = float(effective)
    elif isinstance(effective, str):
        seconds = _parse_duration(effective)
    else:
        raise KeepAliveParseError("keep_alive must be a duration string or seconds")

    if seconds < 0:
        return None

    return seconds


def _parse_duration(value: str) -> float:
    match = _DURATION_RE.match(value)
    if not match:
        raise KeepAliveParseError(
            "keep_alive must be seconds or a duration like 5m, 1h, or 0"
        )

    amount = float(match.group(1))
    unit = match.group(2).lower()
    return amount * _DURATION_MULTIPLIERS[unit]


def release_mlx_memory() -> None:
    gc.collect()

    try:
        import mlx.core as mx
    except ImportError:
        return

    clear_cache = getattr(mx, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
        return

    metal = getattr(mx, "metal", None)
    clear_cache = getattr(metal, "clear_cache", None)
    if callable(clear_cache):
        clear_cache()
