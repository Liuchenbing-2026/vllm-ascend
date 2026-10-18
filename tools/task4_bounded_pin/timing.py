# SPDX-License-Identifier: Apache-2.0
"""Task-only phase limits; no device, process or allocation operations."""

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PhaseLimits:
    preflight_seconds: int = 3 * 60
    entry_seconds: int = 8 * 60
    rank_load_seconds: int = 7 * 60
    all_loaded_seconds: int = 15 * 60
    pin_seconds: int = 3 * 60
    device_seconds: int = 18 * 60


class PhaseTimer:
    def __init__(self, lease, wall, monotonic):
        if not all(math.isfinite(value) for value in (wall, monotonic)):
            raise ValueError("Invalid clock anchor")
        if not lease["start_epoch"] <= wall < lease["start_epoch"] + 60:
            raise ValueError("Timer must be anchored in the first lease minute")
        self.lease = lease
        self.wall = wall
        self.monotonic = monotonic
        self.origin = monotonic - (wall - lease["start_epoch"])
        self.limits = PhaseLimits()
        self.seen = {}
        self.pin_deadline = None

    @property
    def hard_deadline(self):
        return self.origin + self.limits.device_seconds

    def record(self):
        return {
            "lease_id": self.lease["lease_id"],
            "wall_epoch": self.wall,
            "monotonic": self.monotonic,
            "lease_start_monotonic": self.origin,
            "hard_deadline_monotonic": self.hard_deadline,
            "limits_seconds": vars(self.limits),
        }

    def marker(self, output, name, now, rank=None):
        path = output / name
        if not path.exists():
            if name in self.seen:
                raise ValueError("Published marker disappeared: " + name)
            return None
        raw = path.read_bytes()
        if name in self.seen and raw != self.seen[name]:
            raise ValueError("Published marker changed: " + name)
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            # A writer may still be flushing. Missing evidence never resets a
            # deadline, and no successful phase is inferred from partial JSON.
            return None
        stamp = event.get("monotonic")
        if (
            type(stamp) not in (int, float)
            or not math.isfinite(stamp)
            or stamp < self.monotonic
            or event.get("lease_id") != self.lease["lease_id"]
            or event.get("rank") != rank
        ):
            raise ValueError("Invalid timing marker or clock namespace: " + name)
        if stamp > now:
            # A worker can publish after this polling iteration sampled now.
            # Revisit next iteration; this never extends a missing-event limit.
            return None
        self.seen[name] = raw
        return stamp

    def check(self, output, now):
        if not math.isfinite(now) or now < self.monotonic:
            raise ValueError("Invalid monotonic clock")
        if now >= self.hard_deadline:
            raise TimeoutError("Absolute 18 minute device deadline reached")
        limits = self.limits
        preflight = self.marker(output, "preflight_complete.json", now)
        if (now if preflight is None else preflight) >= self.origin + limits.preflight_seconds:
            raise TimeoutError("Deployment/preflight exceeded 3 minutes")
        loaded = []
        completed = []
        for rank in range(8):
            entry = self.marker(output, f"model_entered.rank-{rank}.json", now, rank)
            end = self.marker(output, f"model_loaded.rank-{rank}.json", now, rank)
            done = self.marker(output, f"probe_complete.rank-{rank}.json", now, rank)
            if (now if entry is None else entry) >= self.origin + limits.entry_seconds:
                raise TimeoutError(f"Rank {rank} missing/tardy load entry at minute 8")
            if entry is not None and (preflight is None or entry < preflight):
                raise ValueError("Load entry without preceding preflight")
            if end is not None and (entry is None or end < entry):
                raise ValueError("Load return without preceding entry")
            if entry is not None and (now if end is None else end) - entry >= limits.rank_load_seconds:
                raise TimeoutError(f"Rank {rank} load exceeded 7 minutes")
            if (now if end is None else end) >= self.origin + limits.all_loaded_seconds:
                raise TimeoutError("All-rank loading exceeded minute 15")
            if done is not None and (end is None or done < end):
                raise ValueError("Probe completion without load return")
            loaded.append(end)
            completed.append(done)
        if all(stamp is not None for stamp in loaded):
            pin_start = max(loaded)
            self.pin_deadline = min(pin_start + limits.pin_seconds, self.hard_deadline)
            if any(stamp is not None and stamp < pin_start for stamp in completed):
                raise ValueError("Probe completion before all-rank loading")
            if now - pin_start >= limits.pin_seconds:
                raise TimeoutError("Pin diagnostics exceeded 3 minutes")
            return all(stamp is not None for stamp in completed)
        if any(stamp is not None for stamp in completed):
            raise ValueError("Probe completion while a peer has not loaded")
        return False
