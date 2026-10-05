from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException


class AppError(Exception):
    status_code = 500
    code = "INTERNAL_ERROR"

    def __init__(self, message: str, status_code: int | None = None, code: str | None = None):
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code
        super().__init__(message)


class LLMUnavailableError(AppError):
    status_code = 503
    code = "LLM_UNAVAILABLE"


class LLMBadResponseError(AppError):
    status_code = 502
    code = "LLM_BAD_RESPONSE"


class LLMTimeoutError(AppError):
    status_code = 502
    code = "LLM_TIMEOUT"


class ModelLoadError(AppError):
    status_code = 503
    code = "MODEL_LOAD_FAILED"


def error_payload(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(AppError)
    async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=error_payload(exc.code, exc.message))

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        parts = []
        for err in exc.errors():
            loc = ".".join(str(item) for item in err.get("loc", []) if item != "body")
            msg = err.get("msg", "invalid value")
            parts.append(f"{loc}: {msg}" if loc else msg)
        message = "; ".join(parts) if parts else "Invalid request payload"
        return JSONResponse(status_code=422, content=error_payload("VALIDATION_ERROR", message))

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        codes = {404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}
        code = codes.get(exc.status_code, "HTTP_ERROR")
        return JSONResponse(status_code=exc.status_code, content=error_payload(code, str(exc.detail)))

    @app.exception_handler(Exception)
    async def handle_unhandled_error(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=500, content=error_payload("INTERNAL_ERROR", "An unexpected error occurred"))
