#!/usr/bin/env python3
"""Test script for AsyncHTTPClient."""

import sys
import asyncio

sys.path.insert(0, '/mnt/xtra/openbulma-v4/workspaces/v3-projects')

from async_http_client import (
    AsyncHTTPClient,
    HTTPClientError,
    HTTPConnectionError,
    HTTPTimeoutError,
    HTTPRetryExhaustedError,
    HTTPResponseError
)

print('Testing AsyncHTTPClient...')


async def test_context_manager():
    """Test context manager functionality."""
    print('\n1. Testing context manager...')
    async with AsyncHTTPClient() as client:
        print('   ✓ Client initialized successfully')
        print(f'   ✓ Timeout: {client.timeout.total} seconds')
        print(f'   ✓ Max retries: {client.max_retries}')
    print('   ✓ Client closed successfully')


def test_exceptions():
    """Test exception classes."""
    print('\n2. Testing exception classes...')
    
    try:
        raise HTTPClientError('Test error')
    except HTTPClientError as e:
        print(f'   ✓ HTTPClientError works: {e}')
    
    try:
        raise HTTPConnectionError('Connection failed')
    except HTTPConnectionError as e:
        print(f'   ✓ HTTPConnectionError works: {e}')
    
    try:
        raise HTTPTimeoutError('Request timed out')
    except HTTPTimeoutError as e:
        print(f'   ✓ HTTPTimeoutError works: {e}')
    
    try:
        raise HTTPResponseError('Server error', status_code=500, response_body={'error': 'test'})
    except HTTPResponseError as e:
        print(f'   ✓ HTTPResponseError works: {e} (status: {e.status_code})')
    
    try:
        raise HTTPRetryExhaustedError('Max retries reached')
    except HTTPRetryExhaustedError as e:
        print(f'   ✓ HTTPRetryExhaustedError works: {e}')


async def test_http_methods():
    """Test HTTP methods."""
    print('\n3. Testing HTTP methods...')
    
    async with AsyncHTTPClient() as client:
        # Test GET
        try:
            response = await client.get('https://httpbin.org/get')
            print(f'   ✓ GET request successful (status: {response.get("status", "N/A")})')
        except Exception as e:
            print(f'   ⚠ GET request failed: {e}')
        
        # Test POST with JSON
        try:
            data = {'test': 'data', 'value': 123}
            response = await client.post('https://httpbin.org/post', json=data)
            print(f'   ✓ POST request successful (status: {response.get("status", "N/A")})')
        except Exception as e:
            print(f'   ⚠ POST request failed: {e}')
        
        # Test PUT
        try:
            data = {'updated': 'value'}
            response = await client.put('https://httpbin.org/put', json=data)
            print(f'   ✓ PUT request successful (status: {response.get("status", "N/A")})')
        except Exception as e:
            print(f'   ⚠ PUT request failed: {e}')
        
        # Test PATCH
        try:
            data = {'patched': 'value'}
            response = await client.patch('https://httpbin.org/patch', json=data)
            print(f'   ✓ PATCH request successful (status: {response.get("status", "N/A")})')
        except Exception as e:
            print(f'   ⚠ PATCH request failed: {e}')
        
        # Test DELETE
        try:
            response = await client.delete('https://httpbin.org/delete')
            print(f'   ✓ DELETE request successful (status: {response.get("status", "N/A")})')
        except Exception as e:
            print(f'   ⚠ DELETE request failed: {e}')


async def main():
    """Run all tests."""
    await test_context_manager()
    test_exceptions()
    await test_http_methods()
    print('\n✅ All tests completed!')


if __name__ == '__main__':
    asyncio.run(main())
