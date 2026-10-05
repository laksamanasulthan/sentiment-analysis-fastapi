from fastapi import APIRouter

from app.api.v1.analyze import router as analyze_router
from app.api.v1.local import router as local_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(analyze_router)
api_router.include_router(local_router)
