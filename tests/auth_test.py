from collections.abc import AsyncGenerator

from httpx import AsyncClient
from pydantic import SecretStr
import pytest
import pytest_asyncio

from tests.conftest import DEFAULT_CONFIG, AclientFactory


@pytest_asyncio.fixture()
async def aclient_with_auth(aclient_factory: AclientFactory) -> AsyncGenerator[AsyncClient]:
    """Create a client with API key authentication enabled."""
    config_with_auth = DEFAULT_CONFIG.model_copy(update={"api_key": SecretStr("test-api-key-123")})
    async with aclient_factory(config_with_auth) as aclient:
        yield aclient


@pytest_asyncio.fixture()
async def aclient_without_auth(aclient_factory: AclientFactory) -> AsyncGenerator[AsyncClient]:
    """Create a client without API key authentication."""
    async with aclient_factory(DEFAULT_CONFIG) as aclient:
        yield aclient


@pytest.mark.asyncio
async def test_health_endpoint_public_without_auth(aclient_without_auth: AsyncClient) -> None:
    """Test that /health endpoint is accessible without authentication when auth is disabled."""
    response = await aclient_without_auth.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "OK"}


@pytest.mark.asyncio
async def test_health_endpoint_public_with_auth_enabled(aclient_with_auth: AsyncClient) -> None:
    """Test that /health endpoint is accessible without authentication even when auth is enabled."""
    response = await aclient_with_auth.get("/health")
    assert response.status_code == 200
    assert response.json() == {"message": "OK"}


@pytest.mark.asyncio
async def test_docs_endpoint_public_with_auth_enabled(aclient_with_auth: AsyncClient) -> None:
    """Test that /docs endpoint is accessible without authentication even when auth is enabled."""
    response = await aclient_with_auth.get("/docs")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_openapi_endpoint_public_with_auth_enabled(aclient_with_auth: AsyncClient) -> None:
    """Test that /openapi.json endpoint is accessible without authentication even when auth is enabled."""
    response = await aclient_with_auth.get("/openapi.json")
    assert response.status_code == 200
    assert "openapi" in response.json()


@pytest.mark.asyncio
async def test_protected_endpoint_requires_auth(aclient_with_auth: AsyncClient) -> None:
    """Test that protected endpoints require authentication when auth is enabled."""
    response = await aclient_with_auth.get("/v1/models")
    assert response.status_code == 403
    assert (
        response.json()["detail"]
        == "API key required. Please provide an API key using the Authorization header with Bearer scheme."
    )


@pytest.mark.asyncio
async def test_protected_endpoint_with_correct_api_key(aclient_with_auth: AsyncClient) -> None:
    """Test that protected endpoints accept requests with correct API key."""
    response = await aclient_with_auth.get(
        "/v1/models",
        headers={"Authorization": "Bearer test-api-key-123"},
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_protected_endpoint_with_incorrect_api_key(aclient_with_auth: AsyncClient) -> None:
    """Test that protected endpoints reject requests with incorrect API key."""
    response = await aclient_with_auth.get(
        "/v1/models",
        headers={"Authorization": "Bearer wrong-api-key"},
    )
    assert response.status_code == 403
    assert response.json()["detail"] == "Invalid API key. The provided API key is incorrect."


@pytest.mark.asyncio
async def test_protected_endpoint_without_bearer_prefix(aclient_with_auth: AsyncClient) -> None:
    """Test that protected endpoints reject API keys without Bearer prefix."""
    response = await aclient_with_auth.get(
        "/v1/models",
        headers={"Authorization": "test-api-key-123"},
    )
    assert response.status_code == 403
    assert (
        response.json()["detail"]
        == "API key required. Please provide an API key using the Authorization header with Bearer scheme."
    )


@pytest.mark.asyncio
async def test_protected_endpoint_no_auth_when_disabled(aclient_without_auth: AsyncClient) -> None:
    """Test that protected endpoints work without authentication when auth is disabled."""
    response = await aclient_without_auth.get("/v1/models")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_api_ps_endpoint_requires_auth(aclient_with_auth: AsyncClient) -> None:
    """Test that /api/ps endpoint requires authentication when auth is enabled."""
    response = await aclient_with_auth.get("/api/ps")
    assert response.status_code == 403
    assert "API key required" in response.json()["detail"]


@pytest.mark.asyncio
async def test_api_ps_endpoint_with_correct_auth(aclient_with_auth: AsyncClient) -> None:
    """Test that /api/ps endpoint works with correct authentication."""
    response = await aclient_with_auth.get(
        "/api/ps",
        headers={"Authorization": "Bearer test-api-key-123"},
    )
    assert response.status_code == 200
    assert "models" in response.json()


@pytest.mark.asyncio
async def test_www_authenticate_header_present(aclient_with_auth: AsyncClient) -> None:
    """Test that WWW-Authenticate header is present in 403 responses."""
    response = await aclient_with_auth.get("/v1/models")
    assert response.status_code == 403
    assert "WWW-Authenticate" in response.headers
    assert response.headers["WWW-Authenticate"] == "Bearer"


@pytest.mark.asyncio
async def test_multiple_protected_endpoints_with_auth(aclient_with_auth: AsyncClient) -> None:
    """Test that multiple protected endpoints all require authentication."""
    # Test GET endpoints
    get_endpoints = [
        "/v1/models",
        "/api/ps",
    ]

    for endpoint in get_endpoints:
        # Without auth
        response = await aclient_with_auth.get(endpoint)
        assert response.status_code == 403, f"Endpoint {endpoint} should require auth"

        # With correct auth
        response_with_auth = await aclient_with_auth.get(
            endpoint,
            headers={"Authorization": "Bearer test-api-key-123"},
        )
        assert response_with_auth.status_code == 200, f"Endpoint {endpoint} should accept correct auth"
