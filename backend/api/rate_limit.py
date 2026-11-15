"""Process-local sliding-window limiter for a single Render instance."""
from collections import defaultdict, deque
from threading import Lock
import time
from fastapi import HTTPException, Request


class InMemoryRateLimiter:
    def __init__(self, limit=8, window_seconds=60):
        self.limit = limit
        self.window_seconds = window_seconds
        self._events = defaultdict(deque)
        self._lock = Lock()

    def check(self, request: Request):
        key = request.client.host if request.client else "unknown"
        now = time.monotonic()
        with self._lock:
            events = self._events[key]
            while events and now - events[0] >= self.window_seconds:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = max(1, int(self.window_seconds - (now - events[0])))
                raise HTTPException(429, "Request limit reached. Please wait before trying again.",
                                    headers={"Retry-After": str(retry_after)})
            events.append(now)


limiter = InMemoryRateLimiter()
