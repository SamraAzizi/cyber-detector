from fastapi import Request, HTTPException
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
import logging

class SecurityMiddleware(BaseHTTPMiddleware):

    def __init__(self, app, rate_limit=100):
        super().__init__(app)
        self.rate_limit = rate_limit
        self.logger = logging.getLogger(__name__)



    async def dispatch(self, request: Request, call_next):
        # 1. IP-based rate limiting
        client_ip = request.client.host
        if self._exceeds_rate_limit(client_ip):
            self.logger.warning(f"Rate limit exceeded for {client_ip}")
            raise HTTPException(429, "Too many requests")

        