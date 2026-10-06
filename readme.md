# Analysis Sentiment Using FastAPI

Classify the sentiment of user input with two pluggable engines:

1. **LLM Engine** — `POST /api/v1/analyze` calls any OpenAI-compatible chat completions API (OpenAI, Groq, OpenRouter, Google Gemini, or a custom endpoint). Provider, API key, base URL, and model are all configured from `.env`, plug-and-play.
2. **Local Engine** — `POST /api/v1/analyze/local` runs [`Horizon-Labs/multilingual-sentiment-small`](https://huggingface.co/Horizon-Labs/multilingual-sentiment-small) (quantized ONNX, 256 MB) fully offline on CPU. It supports 35+ languages including Indonesian, no API key required.

## Features

- FastAPI with automatic Swagger UI playground at `/docs`
- Two independent classification engines behind one response contract
- Quantized ONNX inference (~19 ms per request on CPU), no PyTorch dependency
- Structured success and error responses
- Docker setup with model weights baked into the image
- Test suite with mocked engines

## Requirements

- Docker (recommended), or Python 3.12+ for bare metal
- An LLM API key only if you want to use `/api/v1/analyze`; the local engine works without one

## How to Run

### Docker (recommended)

```bash
cp .env.example .env
docker compose up --build -d
```

# Open Swagger UI at `http://localhost:8000/docs`, for Interactive Usage

### Without Docker (Bare Metal)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt -r requirements-ml.txt
uvicorn app.main:app --reload
```

The first call to `/api/v1/analyze/local` downloads the model weights (~289 MB) into your Hugging Face cache. Set `HF_HOME` if you want the cache somewhere specific. Run tests with `pip install -r requirements-dev.txt` then `pytest`.

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | `openai` | `openai`, `groq`, `openrouter`, `gemini`, or `custom` |
| `LLM_API_KEY` | empty | API key for the provider |
| `LLM_BASE_URL` | provider default | Override the chat completions base URL |
| `LLM_MODEL` | provider default | Override the model name |
| `LLM_TIMEOUT_SECONDS` | `30` | LLM request timeout |
| `LOCAL_MODEL_NAME` | `Horizon-Labs/multilingual-sentiment-small` | Hugging Face repo for the local engine |
| `LOCAL_ONNX_FILE` | `onnx/model_quantized.onnx` | ONNX weights inside the repo |
| `LOCAL_MAX_LENGTH` | `1024` | Tokenization truncation limit |

## API

### `GET /health`

```bash
curl http://localhost:8000/health
```

```json
{"status": "ok"}
```

### `POST /api/v1/analyze` (LLM engine)

```bash
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Saya sangat puas dengan layanan pelanggan hari ini, responnya cepat dan ramah!"}'
```

```json
{
  "sentiment": "positive",
  "confidence_score": 0.94,
  "category": "Customer Support",
  "processing_time_ms": 45.2
}
```

`confidence_score` is the LLM's own certainty, `category` is picked from Customer Support, Product Quality, Delivery & Shipping, Pricing & Billing, or General.

### `POST /api/v1/analyze/local` (transformer engine)

Same request and response shape, no API key needed. Real outputs:

```json
{"sentiment": "positive", "confidence_score": 0.9999, "category": "Customer Support", "processing_time_ms": 18.8}
```
```json
{"sentiment": "negative", "confidence_score": 0.9983, "category": "Delivery & Shipping", "processing_time_ms": 19.8}
```
```json
{"sentiment": "neutral", "confidence_score": 0.9999, "category": "General", "processing_time_ms": 19.5}
```

`confidence_score` is the softmax probability of the winning label. The local engine assigns `category` with keyword rules; the LLM engine lets the model choose it.

Prefer Swagger UI over curl: open `http://localhost:8000/docs` and try every endpoint from the browser.

### Errors

All errors share one envelope:

```json
{
  "error": {
    "code": "LLM_UNAVAILABLE",
    "message": "LLM API key is not configured, set LLM_API_KEY in .env"
  }
}
```

| Code | Status | Cause |
| --- | --- | --- |
| `VALIDATION_ERROR` | 422 | Missing, blank, or oversized text (max 2000 chars) |
| `LLM_UNAVAILABLE` | 503 | Missing API key, rejected key, rate limit, unreachable provider |
| `LLM_TIMEOUT` | 502 | Provider did not respond in time |
| `LLM_BAD_RESPONSE` | 502 | Provider replied with something unparseable |
| `MODEL_LOAD_FAILED` | 503 | Local model could not be downloaded or loaded |
| `INTERNAL_ERROR` | 500 | Unexpected failure |

## Project Structure

```
app/
  main.py               app factory, health route, exception handlers
  core/
    config.py           settings from .env, provider presets
    errors.py           error envelope and handlers
  api/v1/
    router.py           v1 route aggregation
    analyze.py          POST /api/v1/analyze
    local.py            POST /api/v1/analyze/local
  schemas/
    analyze.py          request and response models
    common.py           health and error models
  services/
    llm_client.py       OpenAI-compatible client
    sentiment.py        LLM engine orchestration
    local_sentiment.py  ONNX engine (download, tokenize, infer)
    categories.py       keyword category rules
tests/
  conftest.py           fixtures with mocked engines
  test_api.py           endpoint and unit tests
Dockerfile
docker-compose.yml
requirements.txt
requirements-ml.txt
requirements-dev.txt
.env.example
```
