import json
import time
from pathlib import Path
from threading import Lock

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

from app.core.config import get_settings
from app.core.errors import ModelLoadError
from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.services.categories import detect_category

ALLOWED_PATTERNS = ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]

_engine = None
_lock = Lock()


class LocalSentimentEngine:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.tokenizer = None
        self.session = None
        self.labels = []

    def load(self):
        try:
            from huggingface_hub import snapshot_download

            repo_path = Path(
                snapshot_download(
                    repo_id=self.settings.local_model_name,
                    allow_patterns=[self.settings.local_onnx_file, *ALLOWED_PATTERNS],
                )
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.settings.local_model_name)
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(repo_path / self.settings.local_onnx_file),
                sess_options=options,
                providers=["CPUExecutionProvider"],
            )
            config = json.loads((repo_path / "config.json").read_text(encoding="utf-8"))
            self.labels = [config["id2label"][str(index)] for index in range(len(config["id2label"]))]
        except ModelLoadError:
            raise
        except Exception as exc:
            raise ModelLoadError(f"failed to load local sentiment model: {exc}") from exc

    def predict(self, text: str) -> tuple[str, float]:
        encoded = self.tokenizer(text, truncation=True, max_length=self.settings.local_max_length)
        available = {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }
        feed = {
            tensor.name: np.array([available[tensor.name]], dtype=np.int64)
            for tensor in self.session.get_inputs()
            if tensor.name in available
        }
        logits = self.session.run(None, feed)[0][0]
        probabilities = np.exp(logits - np.max(logits))
        probabilities = probabilities / probabilities.sum()
        best = int(np.argmax(probabilities))
        return self.labels[best], float(probabilities[best])


def get_engine() -> LocalSentimentEngine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                engine = LocalSentimentEngine()
                engine.load()
                _engine = engine
    return _engine


class LocalSentimentService:
    def __init__(self, engine=None):
        self.engine = engine

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        engine = self.engine or get_engine()
        started = time.perf_counter()
        sentiment, confidence = engine.predict(request.text)
        elapsed = (time.perf_counter() - started) * 1000
        return AnalyzeResponse(
            sentiment=sentiment,
            confidence_score=round(confidence, 4),
            category=detect_category(request.text),
            processing_time_ms=round(elapsed, 1),
        )
