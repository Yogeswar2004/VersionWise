import logging
from typing import Any

from fastapi import FastAPI
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

from app.config import Settings

logger = logging.getLogger(__name__)


def create_mongo_client(settings: Settings) -> MongoClient[Any] | None:
    """Create a MongoClient instance using the provided settings.

    Returns None if mongodb_uri is empty or not configured.
    """
    if not settings.mongodb_uri.strip():
        return None
    return MongoClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=settings.mongodb_server_selection_timeout_ms,
        connectTimeoutMS=settings.mongodb_connect_timeout_ms,
    )


def ping_mongo_database(client: MongoClient[Any] | None) -> bool:
    """Ping MongoDB cluster to verify connectivity.

    Returns True if ping is successful, or False if client is None or connectivity fails.
    Logs only fixed, sanitized messages. Never logs raw URIs, credentials, or raw exception text.
    """
    if client is None:
        return False
    try:
        client.admin.command("ping")
        return True
    except (ServerSelectionTimeoutError, ConnectionFailure):
        logger.warning("MongoDB ping failed: server selection or connection timeout.")
        return False
    except PyMongoError:
        logger.warning("MongoDB ping failed: database operation error.")
        return False


def get_database(app: FastAPI, settings: Settings) -> Database[Any] | None:
    """Retrieve the configured database from app.state.mongo_client, or None if unavailable."""
    client: MongoClient[Any] | None = getattr(app.state, "mongo_client", None)
    if client is None:
        return None
    return client[settings.mongodb_database]

