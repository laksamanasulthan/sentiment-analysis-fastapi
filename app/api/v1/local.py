from typing import Annotated

from fastapi import APIRouter, Depends

from app.schemas.analyze import AnalyzeRequest, AnalyzeResponse
from app.schemas.common import ErrorEnvelope
from app.services.local_sentiment import LocalSentimentService

router = APIRouter(tags=["analyze"])


def get_local_sentiment_service() -> LocalSentimentService:
    return LocalSentimentService()


@router.post(
    "/analyze/local",
    response_model=AnalyzeResponse,
    responses={503: {"model": ErrorEnvelope}},
)
def analyze_local_sentiment(
    request: AnalyzeRequest,
    service: Annotated[LocalSentimentService, Depends(get_local_sentiment_service)],
) -> AnalyzeResponse:
    return service.analyze(request)
