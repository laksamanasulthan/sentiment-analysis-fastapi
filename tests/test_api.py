import pytest

from app.schemas.analyze import AnalyzeRequest
from app.services.categories import detect_category
from app.services.llm_client import LLMBadResponseError, LLMClient
from app.services.local_sentiment import LocalSentimentService


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_analyze_positive(client):
    response = client.post("/api/v1/analyze", json={"text": "great service, fast and friendly"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"sentiment", "confidence_score", "category", "processing_time_ms"}
    assert body["sentiment"] == "positive"
    assert 0.0 <= body["confidence_score"] <= 1.0
    assert body["processing_time_ms"] >= 0


def test_analyze_blank_text_returns_validation_error(client):
    response = client.post("/api/v1/analyze", json={"text": "   "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_analyze_missing_text_returns_validation_error(client):
    response = client.post("/api/v1/analyze", json={})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_analyze_text_too_long_returns_validation_error(client):
    response = client.post("/api/v1/analyze", json={"text": "a" * 2001})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_analyze_without_api_key(client_no_key):
    response = client_no_key.post("/api/v1/analyze", json={"text": "hello there"})
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "LLM_UNAVAILABLE"


def test_analyze_local_positive(client):
    response = client.post("/api/v1/analyze/local", json={"text": "Layanan pelanggannya cepat dan ramah"})
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"sentiment", "confidence_score", "category", "processing_time_ms"}
    assert body["sentiment"] == "positive"


def test_analyze_local_blank_text_returns_validation_error(client):
    response = client.post("/api/v1/analyze/local", json={"text": ""})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "VALIDATION_ERROR"


def test_local_service_maps_engine_result():
    class StubEngine:
        def predict(self, text: str) -> tuple[str, float]:
            return "negative", 0.87

    response = LocalSentimentService(StubEngine()).analyze(
        AnalyzeRequest(text="Barangnya rusak dan kualitasnya jelek")
    )
    assert response.sentiment == "negative"
    assert response.confidence_score == 0.87
    assert response.category == "Product Quality"
    assert response.processing_time_ms >= 0


def test_llm_parse_extracts_json_object():
    result = LLMClient()._parse(
        'Sure! {"sentiment": "negative", "confidence": 0.8, "category": "Product Quality"}'
    )
    assert result == {"sentiment": "negative", "confidence": 0.8, "category": "Product Quality"}


def test_llm_parse_rejects_unknown_sentiment():
    with pytest.raises(LLMBadResponseError):
        LLMClient()._parse('{"sentiment": "angry", "confidence": 0.9, "category": "General"}')


def test_detect_category_support():
    assert (
        detect_category("Saya sangat puas dengan layanan pelanggan hari ini, responnya cepat dan ramah!")
        == "Customer Support"
    )


def test_detect_category_delivery():
    assert detect_category("Paket datang terlambat tiga hari") == "Delivery & Shipping"


def test_detect_category_pricing():
    assert detect_category("Harganya mahal tapi bayarnya susah") == "Pricing & Billing"


def test_detect_category_general():
    assert detect_category("Hari ini cuaca cerah sekali") == "General"
