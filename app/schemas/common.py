from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str = "ok"


class ErrorResponse(BaseModel):
    code: str
    message: str


class ErrorEnvelope(BaseModel):
    error: ErrorResponse
