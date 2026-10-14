import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response, status
from pymongo.errors import PyMongoError

from app.config import settings
from app.database import create_mongo_client, ping_mongo_database
from app.retrieval import VectorSearchUnavailableError, perform_vector_search
from app.schemas import RetrievalQuery, RetrievalResponse

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.mongodb_uri.strip():
        try:
            client = create_mongo_client(settings)
            app.state.mongo_client = client
            if ping_mongo_database(client):
                logger.info("MongoDB client initialized and connection verified.")
            else:
                logger.warning(
                    "MongoDB configured but unreachable during startup. Starting in degraded mode."
                )
        except PyMongoError:
            app.state.mongo_client = None
            logger.warning(
                "MongoDB client initialization failed. Starting in degraded mode."
            )
    else:
        app.state.mongo_client = None
        logger.info("MongoDB unconfigured. Starting without database client.")

    try:
        yield
    finally:
        active_client = getattr(app.state, "mongo_client", None)
        if active_client is not None:
            active_client.close()
            logger.info("MongoDB connection pool closed.")


app = FastAPI(title=settings.app_name, lifespan=lifespan)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/health/db")
def health_db_check(response: Response, request: Request) -> dict[str, str]:
    client = getattr(request.app.state, "mongo_client", None)
    if client is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable"}

    if not ping_mongo_database(client):
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "unavailable"}

    return {"status": "connected"}


@app.post("/api/v1/retrieval/search", response_model=RetrievalResponse)
def search_documentation(query: RetrievalQuery, request: Request) -> RetrievalResponse:
    client = getattr(request.app.state, "mongo_client", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database connection is unavailable.",
        )
    db = client[settings.mongodb_database]
    try:
        return perform_vector_search(db=db, query=query, settings=settings)
    except VectorSearchUnavailableError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Vector search service unavailable.",
        )
