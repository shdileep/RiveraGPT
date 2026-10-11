import pytest
from fastapi.testclient import TestClient
from app import app


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "RiveraGPT" in data["engine"]
    assert data["shield"] == "active"


def test_models_catalog_endpoint(client):
    response = client.get("/models")
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    keys = [m["key"] for m in data["models"]]
    assert "auto" in keys
    assert "fast" in keys


def test_chat_icon_endpoint(client):
    response = client.get("/chat.png")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"


def test_rename_validation(client):
    # Empty title rejected
    res_empty = client.put("/conversations/nonexistent/rename", json={"title": "   "})
    assert res_empty.status_code == 400

    # Invalid body rejected
    res_invalid = client.put(
        "/conversations/nonexistent/rename",
        data="not a json",
        headers={"Content-Type": "application/json"}
    )
    assert res_invalid.status_code == 400


def test_stream_chat_validation(client):
    # Missing message
    res = client.post("/chat/stream", json={"message": "   "})
    assert res.status_code == 400


def test_upload_document(client):
    res = client.post(
        "/upload",
        files={"file": ("notes.txt", b"Summary of Q4 performance", "text/plain")},
        data={"thread_id": "test_thread_upload"}
    )
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["filename"] == "notes.txt"

