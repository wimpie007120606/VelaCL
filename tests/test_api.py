from fastapi.testclient import TestClient

from apps.api.main import app, detect_language


def test_health_is_available_without_model():
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_language_heuristic_avoids_ambiguous_english_marker():
    assert detect_language("my card payment failed") == "en"
    assert detect_language("sawubona, ngidinga usizo") == "zu"
