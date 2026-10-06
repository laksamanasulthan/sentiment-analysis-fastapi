from pydantic import BaseModel, Field, field_validator


class AnalyzeRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000, examples=["Saya sangat puas dengan layanan pelanggan hari ini, responnya cepat dan ramah!"])

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be blank")
        return stripped


class AnalyzeResponse(BaseModel):
    sentiment: str
    confidence_score: float
    category: str
    processing_time_ms: float
