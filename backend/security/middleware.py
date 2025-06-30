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

        
        # 2. Block suspicious user agents
        if "sqlmap" in request.headers.get("user-agent", "").lower():
            self.logger.warning(f"Blocked malicious UA from {client_ip}")
            raise HTTPException(403, "Forbidden")

        response = await call_next(request)
        

        # 3. Security headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "Content-Security-Policy": "default-src 'self'"
        })
        
        return response
    
    

    def _exceeds_rate_limit(self, ip: str) -> bool:
        # Implement Redis-based counter in production
        return False  # Placeholder

# In app.py:
# app.add_middleware(SecurityMiddleware)
# app.add_middleware(HTTPSRedirectMiddleware)

        