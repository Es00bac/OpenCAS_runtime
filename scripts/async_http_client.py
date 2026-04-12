"""
Async HTTP Client using aiohttp with advanced features.

This module provides a robust async HTTP client with:
- Connection pooling
- Automatic retries with exponential backoff
- Configurable timeouts
- Async context manager support
- Comprehensive error handling
- Logging support
"""

import asyncio
import logging
from typing import Any, Dict, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import random

import aiohttp
from aiohttp import ClientSession, ClientResponse, ClientTimeout


# Configure logging
logger = logging.getLogger(__name__)


class HTTPMethod(Enum):
    """Supported HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"


class HTTPClientError(Exception):
    """Base exception for HTTP client errors."""
    pass


class HTTPConnectionError(HTTPClientError):
    """Raised when a connection error occurs."""
    pass


class HTTPTimeoutError(HTTPClientError):
    """Raised when a request times out."""
    pass


class HTTPResponseError(HTTPClientError):
    """Raised when the server returns an error response."""
    
    def __init__(self, message: str, status_code: int = None, response_body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class HTTPRetryExhaustedError(HTTPClientError):
    """Raised when all retry attempts have been exhausted."""
    pass


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_base: float = 2.0
    retry_on_status_codes: tuple = (500, 502, 503, 504)
    retry_on_exceptions: tuple = (aiohttp.ClientConnectionError, asyncio.TimeoutError)


@dataclass
class TimeoutConfig:
    """Configuration for request timeouts."""
    total: float = 30.0
    connect: float = 10.0
    sock_read: float = 30.0
    sock_connect: float = 10.0


class AsyncHTTPClient:
    """
    Async HTTP client with connection pooling, retries, and comprehensive error handling.
    
    Features:
    - Connection pooling via aiohttp
    - Automatic retries with exponential backoff
    - Configurable timeouts
    - Async context manager support
    - Support for common HTTP methods
    - Comprehensive logging
    
    Example:
        async with AsyncHTTPClient() as client:
            response = await client.get("https://api.example.com/data")
            data = await response.json()
    """
    
    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout_config: Optional[TimeoutConfig] = None,
        retry_config: Optional[RetryConfig] = None,
        headers: Optional[Dict[str, str]] = None,
        pool_size: int = 100,
        enable_logging: bool = True
    ):
        """
        Initialize the async HTTP client.
        
        Args:
            base_url: Base URL for all requests
            timeout_config: Timeout configuration
            retry_config: Retry configuration
            headers: Default headers for all requests
            pool_size: Connection pool size
            enable_logging: Whether to enable logging
        """
        self.base_url = base_url.rstrip('/') if base_url else None
        self.timeout_config = timeout_config or TimeoutConfig()
        self.retry_config = retry_config or RetryConfig()
        self.default_headers = headers or {}
        self.pool_size = pool_size
        self.enable_logging = enable_logging
        
        self._session: Optional[ClientSession] = None
        self._connector: Optional[aiohttp.TCPConnector] = None
        self._closed = True
        
        if self.enable_logging:
            self._setup_logging()
    
    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
    
    def _get_timeout(self) -> ClientTimeout:
        """Create aiohttp ClientTimeout from config."""
        return ClientTimeout(
            total=self.timeout_config.total,
            connect=self.timeout_config.connect,
            sock_read=self.timeout_config.sock_read,
            sock_connect=self.timeout_config.sock_connect
        )
    
    def _create_connector(self) -> aiohttp.TCPConnector:
        """Create TCP connector with connection pooling."""
        return aiohttp.TCPConnector(
            limit=self.pool_size,
            limit_per_host=10,
            enable_cleanup_closed=True,
            force_close=False,
        )
    
    async def __aenter__(self) -> 'AsyncHTTPClient':
        """Async context manager entry."""
        await self.open()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    async def open(self) -> None:
        """Open the client session."""
        if self._session is not None and not self._session.closed:
            return
        
        self._connector = self._create_connector()
        timeout = self._get_timeout()
        
        self._session = ClientSession(
            connector=self._connector,
            timeout=timeout,
            headers=self.default_headers,
            raise_for_status=False
        )
        self._closed = False
        
        if self.enable_logging:
            logger.info("AsyncHTTPClient session opened")
    
    async def close(self) -> None:
        """Close the client session."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None
        
        if self._connector and not self._connector.closed:
            await self._connector.close()
            self._connector = None
        
        self._closed = True
        
        if self.enable_logging:
            logger.info("AsyncHTTPClient session closed")
    
    def _build_url(self, url: str) -> str:
        """Build full URL from base URL and endpoint."""
        if self.base_url and not url.startswith(('http://', 'https://')):
            return f"{self.base_url}/{url.lstrip('/')}"
        return url
    
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff and jitter."""
        delay = min(
            self.retry_config.base_delay * (self.retry_config.exponential_base ** attempt),
            self.retry_config.max_delay
        )
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0, delay * 0.1)
        return delay + jitter
    
    async def _make_request(
        self,
        method: HTTPMethod,
        url: str,
        **kwargs
    ) -> ClientResponse:
        """
        Make a single HTTP request.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional arguments for aiohttp request
            
        Returns:
            ClientResponse object
            
        Raises:
            HTTPConnectionError: On connection errors
            HTTPTimeoutError: On timeout errors
        """
        if self._session is None or self._session.closed:
            raise HTTPClientError("Client session is not open. Use 'async with' or call open()")
        
        full_url = self._build_url(url)
        
        try:
            if self.enable_logging:
                logger.debug(f"Making {method.value} request to {full_url}")
            
            async with self._session.request(
                method.value,
                full_url,
                **kwargs
            ) as response:
                await response.read()  # Read response to allow connection reuse
                return response
                
        except asyncio.TimeoutError as e:
            if self.enable_logging:
                logger.error(f"Request timeout for {full_url}: {e}")
            raise HTTPTimeoutError(f"Request timeout: {e}") from e
        except aiohttp.ClientConnectionError as e:
            if self.enable_logging:
                logger.error(f"Connection error for {full_url}: {e}")
            raise HTTPConnectionError(f"Connection error: {e}") from e
        except Exception as e:
            if self.enable_logging:
                logger.error(f"Unexpected error for {full_url}: {e}")
            raise HTTPClientError(f"Unexpected error: {e}") from e
    
    async def _request_with_retry(
        self,
        method: HTTPMethod,
        url: str,
        **kwargs
    ) -> ClientResponse:
        """
        Make HTTP request with automatic retries.
        
        Args:
            method: HTTP method
            url: Request URL
            **kwargs: Additional arguments for aiohttp request
            
        Returns:
            ClientResponse object
            
        Raises:
            HTTPRetryExhaustedError: When all retries are exhausted
        """
        last_exception = None
        
        for attempt in range(self.retry_config.max_retries + 1):
            try:
                response = await self._make_request(method, url, **kwargs)
                
                # Check if we should retry based on status code
                if response.status in self.retry_config.retry_on_status_codes:
                    if attempt < self.retry_config.max_retries:
                        delay = self._calculate_retry_delay(attempt)
                        if self.enable_logging:
                            logger.warning(
                                f"Got status {response.status}, retrying in {delay:.2f}s "
                                f"(attempt {attempt + 1}/{self.retry_config.max_retries})"
                            )
                        await asyncio.sleep(delay)
                        continue
                
                return response
                
            except self.retry_config.retry_on_exceptions as e:
                last_exception = e
                if attempt < self.retry_config.max_retries:
                    delay = self._calculate_retry_delay(attempt)
                    if self.enable_logging:
                        logger.warning(
                            f"Request failed: {e}, retrying in {delay:.2f}s "
                            f"(attempt {attempt + 1}/{self.retry_config.max_retries})"
                        )
                    await asyncio.sleep(delay)
                else:
                    break
        
        # All retries exhausted
        error_msg = f"All {self.retry_config.max_retries} retry attempts exhausted"
        if self.enable_logging:
            logger.error(error_msg)
        raise HTTPRetryExhaustedError(error_msg) from last_exception
    
    async def request(
        self,
        method: Union[str, HTTPMethod],
        url: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ClientResponse:
        """
        Make an HTTP request with the specified method.
        
        Args:
            method: HTTP method (string or HTTPMethod enum)
            url: Request URL
            params: Query parameters
            data: Request body data
            json: JSON request body
            headers: Additional headers
            **kwargs: Additional arguments for aiohttp
            
        Returns:
            ClientResponse object
        """
        if isinstance(method, str):
            method = HTTPMethod(method.upper())
        
        request_headers = {**self.default_headers}
        if headers:
            request_headers.update(headers)
        
        return await self._request_with_retry(
            method,
            url,
            params=params,
            data=data,
            json=json,
            headers=request_headers,
            **kwargs
        )
    
    async def get(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ClientResponse:
        """Make a GET request."""
        return await self.request(
            HTTPMethod.GET, url, params=params, headers=headers, **kwargs
        )
    
    async def post(
        self,
        url: str,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ClientResponse:
        """Make a POST request."""
        return await self.request(
            HTTPMethod.POST, url, data=data, json=json, 
            params=params, headers=headers, **kwargs
        )
    
    async def put(
        self,
        url: str,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ClientResponse:
        """Make a PUT request."""
        return await self.request(
            HTTPMethod.PUT, url, data=data, json=json,
            params=params, headers=headers, **kwargs
        )
    
    async def delete(
        self,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ClientResponse:
        """Make a DELETE request."""
        return await self.request(
            HTTPMethod.DELETE, url, params=params, headers=headers, **kwargs
        )
    
    async def patch(
        self,
        url: str,
        data: Optional[Any] = None,
        json: Optional[Any] = None,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        **kwargs
    ) -> ClientResponse:
        """Make a PATCH request."""
        return await self.request(
            HTTPMethod.PATCH, url, data=data, json=json,
            params=params, headers=headers, **kwargs
        )
    
    @property
    def closed(self) -> bool:
        """Check if the client is closed."""
        return self._closed


# Convenience functions for simple use cases

async def get(
    url: str,
    params: Optional[Dict[str, Any]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ClientResponse:
    """Simple GET request with automatic session management."""
    timeout_config = TimeoutConfig(total=timeout)
    async with AsyncHTTPClient(timeout_config=timeout_config) as client:
        return await client.get(url, params=params, headers=headers, **kwargs)


async def post(
    url: str,
    data: Optional[Any] = None,
    json: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ClientResponse:
    """Simple POST request with automatic session management."""
    timeout_config = TimeoutConfig(total=timeout)
    async with AsyncHTTPClient(timeout_config=timeout_config) as client:
        return await client.post(url, data=data, json=json, headers=headers, **kwargs)


async def put(
    url: str,
    data: Optional[Any] = None,
    json: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ClientResponse:
    """Simple PUT request with automatic session management."""
    timeout_config = TimeoutConfig(total=timeout)
    async with AsyncHTTPClient(timeout_config=timeout_config) as client:
        return await client.put(url, data=data, json=json, headers=headers, **kwargs)


async def delete(
    url: str,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ClientResponse:
    """Simple DELETE request with automatic session management."""
    timeout_config = TimeoutConfig(total=timeout)
    async with AsyncHTTPClient(timeout_config=timeout_config) as client:
        return await client.delete(url, headers=headers, **kwargs)


async def patch(
    url: str,
    data: Optional[Any] = None,
    json: Optional[Any] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: float = 30.0,
    **kwargs
) -> ClientResponse:
    """Simple PATCH request with automatic session management."""
    timeout_config = TimeoutConfig(total=timeout)
    async with AsyncHTTPClient(timeout_config=timeout_config) as client:
        return await client.patch(url, data=data, json=json, headers=headers, **kwargs)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def example():
        """Example usage of the async HTTP client."""
        # Configure retry and timeout
        retry_config = RetryConfig(
            max_retries=3,
            base_delay=1.0,
            retry_on_status_codes=(500, 502, 503, 504)
        )
        
        timeout_config = TimeoutConfig(
            total=30.0,
            connect=10.0
        )
        
        # Using context manager (recommended)
        async with AsyncHTTPClient(
            retry_config=retry_config,
            timeout_config=timeout_config,
            headers={"User-Agent": "AsyncHTTPClient/1.0"}
        ) as client:
            try:
                # GET request
                response = await client.get("https://httpbin.org/get")
                print(f"GET Status: {response.status}")
                data = await response.json()
                print(f"GET Response: {data}")
                
                # POST request with JSON
                response = await client.post(
                    "https://httpbin.org/post",
                    json={"key": "value", "number": 42}
                )
                print(f"POST Status: {response.status}")
                
                # PUT request
                response = await client.put(
                    "https://httpbin.org/put",
                    json={"updated": True}
                )
                print(f"PUT Status: {response.status}")
                
                # DELETE request
                response = await client.delete("https://httpbin.org/delete")
                print(f"DELETE Status: {response.status}")
                
            except HTTPTimeoutError as e:
                print(f"Request timed out: {e}")
            except HTTPConnectionError as e:
                print(f"Connection error: {e}")
            except HTTPRetryExhaustedError as e:
                print(f"All retries exhausted: {e}")
            except HTTPClientError as e:
                print(f"HTTP client error: {e}")
        
        # Simple one-off request
        response = await get("https://httpbin.org/get")
        print(f"Simple GET Status: {response.status}")
    
    # Run the example
    asyncio.run(example())
