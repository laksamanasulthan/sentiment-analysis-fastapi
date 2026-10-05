FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt requirements-ml.txt ./
RUN pip install -r requirements.txt -r requirements-ml.txt

COPY app ./app

RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Horizon-Labs/multilingual-sentiment-small', allow_patterns=['onnx/model_quantized.onnx', 'config.json', 'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json'])"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
