from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.schemas.common import ErrorEnvelope
from app.services.sentiment import SentimentService

router = APIRouter(tags=["analyze"])


def get_sentiment_service() -> SentimentService:
    return SentimentService()


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={502: {"model": ErrorEnvelope}, 503: {"model": ErrorEnvelope}},
)
def analyze_sentiment(
    request: AnalyzeRequest,
    service: Annotated[SentimentService, Depends(get_sentiment_service)],
) -> AnalyzeResponse:
    return service.analyze(request)
