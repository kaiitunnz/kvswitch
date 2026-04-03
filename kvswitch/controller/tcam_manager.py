"""Admission and eviction policy for KVSwitch TCAM-backed prefix rules."""

import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Any

PrefixKey = tuple[int, ...]


@dataclass
class InstalledPrefixRule:
    """An installed prefix rule tracked by the control plane."""

    prefix: PrefixKey
    target_id: str
    hit_count: int
    last_used: float


class TcamManager:
    """Frequency-threshold admission with sliding-window counting and LRU eviction."""

    def __init__(
        self,
        admission_threshold: int = 2,
        window_s: float = 60.0,
        max_entries: int = 1024,
    ) -> None:
        if admission_threshold <= 0:
            raise ValueError("admission_threshold must be positive")
        if window_s <= 0:
            raise ValueError("window_s must be positive")
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")

        self.admission_threshold = admission_threshold
        self.window_s = window_s
        self.max_entries = max_entries
        self._observations: dict[PrefixKey, deque[float]] = {}
        self._installed: OrderedDict[PrefixKey, InstalledPrefixRule] = OrderedDict()

    def _trim_window(self, prefix: PrefixKey, now: float) -> deque[float]:
        window = self._observations.setdefault(prefix, deque())
        cutoff = now - self.window_s
        while window and window[0] < cutoff:
            window.popleft()
        if not window:
            self._observations.pop(prefix, None)
        return window

    def record_observation(self, prefix: PrefixKey, now: float | None = None) -> int:
        now = time.time() if now is None else now
        window = self._observations.setdefault(prefix, deque())
        window.append(now)
        self._trim_window(prefix, now)
        return len(self._observations.get(prefix, ()))

    def observation_count(self, prefix: PrefixKey, now: float | None = None) -> int:
        now = time.time() if now is None else now
        return len(self._trim_window(prefix, now))

    def admitted(self, prefix: PrefixKey, now: float | None = None) -> bool:
        return self.observation_count(prefix, now) >= self.admission_threshold

    def installed_rule(self, prefix: PrefixKey) -> InstalledPrefixRule | None:
        return self._installed.get(prefix)

    def touch(self, prefix: PrefixKey, now: float | None = None) -> None:
        rule = self._installed.get(prefix)
        if rule is None:
            return
        rule.last_used = time.time() if now is None else now
        self._installed.move_to_end(prefix)

    def install(
        self,
        prefix: PrefixKey,
        target_id: str,
        hit_count: int,
        now: float | None = None,
    ) -> tuple[InstalledPrefixRule, tuple[PrefixKey, InstalledPrefixRule] | None]:
        now = time.time() if now is None else now
        evicted: tuple[PrefixKey, InstalledPrefixRule] | None = None

        existing = self._installed.get(prefix)
        if existing is not None:
            existing.target_id = target_id
            existing.hit_count = hit_count
            existing.last_used = now
            self._installed.move_to_end(prefix)
            return existing, None

        if len(self._installed) >= self.max_entries:
            evicted = self._installed.popitem(last=False)

        rule = InstalledPrefixRule(
            prefix=prefix,
            target_id=target_id,
            hit_count=hit_count,
            last_used=now,
        )
        self._installed[prefix] = rule
        return rule, evicted

    def remove(self, prefix: PrefixKey) -> InstalledPrefixRule | None:
        return self._installed.pop(prefix, None)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {
            ".".join(f"{value:08x}" for value in prefix): {
                "target_id": rule.target_id,
                "hit_count": rule.hit_count,
                "last_used": rule.last_used,
            }
            for prefix, rule in self._installed.items()
        }
