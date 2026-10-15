from __future__ import annotations

from email.utils import parsedate_to_datetime
import random
import time as system_time

from .models import BudgetExhausted, CollectionPaused


class Clock:
    time = staticmethod(system_time.time)
    monotonic = staticmethod(system_time.monotonic)
    sleep = staticmethod(system_time.sleep)


def retry_after_seconds(value: str | None, now: float) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - now)
        except (ValueError, TypeError, OverflowError):
            return None


class RateLimiter:
    def __init__(self, store, policy, run_id, clock=None):
        self.store, self.policy, self.run_id = store, policy, run_id
        self.clock = clock or Clock()
        self.last_monotonic = None

    def sleep(self, seconds):
        end = self.clock.monotonic() + max(0, seconds)
        while (remaining := end - self.clock.monotonic()) > 0:
            self.clock.sleep(min(remaining, 1))

    def reserve(self, job):
        if self.store.setting("access_block"):
            raise CollectionPaused("Access block is latched. Resolve access, then explicitly clear the block.")
        now = self.clock.time()
        cooldown = self.store.setting("cooldown_until", 0)
        if cooldown > now:
            raise CollectionPaused(f"Shared cooldown remains active until UTC epoch {cooldown}")
        if self.policy.max_attempts_per_run is not None:
            count = self.store.db.execute("SELECT COUNT(*) FROM attempts WHERE run_id=?", (self.run_id,)).fetchone()[0]
            if count >= self.policy.max_attempts_per_run:
                raise BudgetExhausted("Run attempt budget exhausted; resume from the checkpoint")
        while True:
            now = self.clock.time()
            rows = self.store.db.execute("SELECT started FROM attempts WHERE started>? ORDER BY started",
                                         (now-3600,)).fetchall() if any(v is not None for v in (
                                             self.policy.min_interval_seconds,self.policy.per_minute,self.policy.per_hour)) else []
            times = [r[0] for r in rows]
            waits = [0.0]
            finished=self.store.setting("last_request_finished")
            if finished is not None and self.policy.min_interval_seconds is not None:
                waits.append(finished+self.policy.min_interval_seconds-now)
            if times and self.policy.min_interval_seconds is not None:
                waits.append(times[-1] + self.policy.min_interval_seconds - now)
            if self.last_monotonic is not None and self.policy.min_interval_seconds is not None:
                waits.append(self.last_monotonic + self.policy.min_interval_seconds - self.clock.monotonic())
            minute = [t for t in times if t > now-60]
            if self.policy.per_minute is not None and len(minute) >= self.policy.per_minute:
                waits.append(minute[-self.policy.per_minute]+60-now)
            if self.policy.per_hour is not None and len(times) >= self.policy.per_hour:
                waits.append(times[-self.policy.per_hour]+3600-now)
            wait = max(waits)
            if wait <= 0:
                break
            self.sleep(wait)
        # One new draw per outgoing attempt, after mandatory pacing. Cached
        # components never reach reserve(); retries do. The first request in
        # a fresh store has no preceding request and needs no added delay.
        maximum = self.policy.random_delay_max_seconds
        if maximum and (self.last_monotonic is not None or
                        self.store.setting("last_request_finished") is not None):
            self.sleep(random.uniform(0.0, maximum))
        with self.store.db:
            cur = self.store.db.execute("INSERT INTO attempts(run_id,key,started) VALUES (?,?,?)",
                                        (self.run_id,job.key,self.clock.time()))
        self.last_monotonic = self.clock.monotonic()
        return cur.lastrowid

    def complete(self, attempt_id, status, outcome):
        with self.store.db:
            self.store.db.execute("UPDATE attempts SET status=?,outcome=? WHERE id=?",
                                  (status,outcome,attempt_id))
        # Spacing from completion is conservative and also covers a slow first
        # browser startup between reservation and actual HTTP dispatch.
        self.last_monotonic=self.clock.monotonic()
        self.store.set_setting("last_request_finished",self.clock.time())

    def cooldown(self, seconds):
        until = max(self.store.setting("cooldown_until", 0), self.clock.time()+seconds)
        self.store.set_setting("cooldown_until", until)
