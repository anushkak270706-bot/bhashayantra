"""Simple per-visitor rate limit for POST requests (no extra packages).

Keeps the free server usable if someone hammers it. In-memory, so it resets
when the server restarts, which is fine for a single small instance.
"""
import time
from collections import defaultdict, deque

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, limit: int = 30, window: int = 60):
        super().__init__(app)
        self.limit, self.window = limit, window
        self.hits = defaultdict(deque)

    def _visitor(self, request) -> str:
        # Render sits behind a proxy: the real visitor IP is the first X-Forwarded-For entry
        fwd = request.headers.get("x-forwarded-for")
        if fwd:
            return fwd.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    async def dispatch(self, request, call_next):
        if request.method == "POST":
            now = time.monotonic()
            q = self.hits[self._visitor(request)]
            while q and now - q[0] > self.window:
                q.popleft()
            if len(q) >= self.limit:
                return JSONResponse(
                    {"detail": "Too many requests. Please wait a minute and try again."},
                    status_code=429, headers={"Retry-After": str(self.window)})
            q.append(now)
            if len(self.hits) > 5000:  # forget idle visitors so memory stays small
                for ip in [ip for ip, d in self.hits.items() if not d or now - d[-1] > self.window]:
                    del self.hits[ip]
        return await call_next(request)
