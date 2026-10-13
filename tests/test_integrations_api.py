import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_integrations_status_endpoint(client):
    res = client.get("/api/integrations/status")
    assert res.status_code == 200
    data = res.json()
    assert "integrations" in data
    items = data["integrations"]
    ids = [i["id"] for i in items]
    assert "google" in ids
    assert "openrouter" in ids
    assert "huggingface" in ids
    assert "flux_image" in ids
    assert "weatherapi" in ids
    assert "openmeteo" in ids
    assert "gnews" in ids
    assert "adzuna" in ids
    assert "heigit" in ids


def test_integrations_test_endpoint(client):
    # Test zero-key Open-Meteo weather ping
    res = client.post("/api/integrations/test", json={"provider": "openmeteo"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "latency_ms" in data


def test_integrations_save_endpoint(client):
    # Test saving without breaking existing environment
    res = client.post("/api/integrations/save", json={"keys": {"TEST_CONFIG_KEY": "test_value_123"}})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "TEST_CONFIG_KEY" in data["saved_keys"]
