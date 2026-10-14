import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
from pymongo.errors import ConnectionFailure, PyMongoError, ServerSelectionTimeoutError

from app import main as app_main
from app.config import Settings
from app.database import create_mongo_client, get_database, ping_mongo_database
from app.main import app


def test_config_defaults() -> None:
    s = Settings()
    assert s.mongodb_uri == ""
    assert s.mongodb_database == "versionwise"
    assert s.mongodb_server_selection_timeout_ms == 2000
    assert s.mongodb_connect_timeout_ms == 2000


def test_config_custom_values() -> None:
    s = Settings(
        mongodb_uri="mongodb://localhost:27017",
        mongodb_database="custom_db",
        mongodb_server_selection_timeout_ms=5000,
        mongodb_connect_timeout_ms=4000,
    )
    assert s.mongodb_uri == "mongodb://localhost:27017"
    assert s.mongodb_database == "custom_db"
    assert s.mongodb_server_selection_timeout_ms == 5000
    assert s.mongodb_connect_timeout_ms == 4000


def test_config_timeout_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(mongodb_server_selection_timeout_ms=0)
    with pytest.raises(ValidationError):
        Settings(mongodb_server_selection_timeout_ms=-100)
    with pytest.raises(ValidationError):
        Settings(mongodb_server_selection_timeout_ms=60001)
    with pytest.raises(ValidationError):
        Settings(mongodb_connect_timeout_ms=0)
    with pytest.raises(ValidationError):
        Settings(mongodb_connect_timeout_ms=60001)


def test_config_database_name_validation() -> None:
    with pytest.raises(ValidationError):
        Settings(mongodb_database="")
    with pytest.raises(ValidationError):
        Settings(mongodb_database="   ")


def test_create_mongo_client_empty_uri() -> None:
    assert create_mongo_client(Settings(mongodb_uri="")) is None
    assert create_mongo_client(Settings(mongodb_uri="   ")) is None


def test_create_mongo_client_with_uri() -> None:
    with patch("app.database.MongoClient") as mock_mongo_cls:
        s = Settings(
            mongodb_uri="mongodb://localhost:27017",
            mongodb_server_selection_timeout_ms=3000,
            mongodb_connect_timeout_ms=4000,
        )
        client = create_mongo_client(s)
        mock_mongo_cls.assert_called_once_with(
            "mongodb://localhost:27017",
            serverSelectionTimeoutMS=3000,
            connectTimeoutMS=4000,
        )
        assert client == mock_mongo_cls.return_value


def test_ping_mongo_database_none_client() -> None:
    assert ping_mongo_database(None) is False


def test_ping_mongo_database_success() -> None:
    mock_client = MagicMock()
    mock_client.admin.command.return_value = {"ok": 1}
    assert ping_mongo_database(mock_client) is True
    mock_client.admin.command.assert_called_once_with("ping")


def test_ping_mongo_database_server_timeout() -> None:
    mock_client = MagicMock()
    mock_client.admin.command.side_effect = ServerSelectionTimeoutError("timed out")
    assert ping_mongo_database(mock_client) is False


def test_ping_mongo_database_connection_failure() -> None:
    mock_client = MagicMock()
    mock_client.admin.command.side_effect = ConnectionFailure("connection dropped")
    assert ping_mongo_database(mock_client) is False


def test_ping_mongo_database_general_pymongo_error() -> None:
    mock_client = MagicMock()
    mock_client.admin.command.side_effect = PyMongoError("database error")
    assert ping_mongo_database(mock_client) is False


def test_ping_mongo_database_programming_error_not_swallowed() -> None:
    mock_client = MagicMock()
    mock_client.admin.command.side_effect = TypeError("unexpected type error")
    with pytest.raises(TypeError):
        ping_mongo_database(mock_client)


def test_get_database() -> None:
    test_app = FastAPI()
    s = Settings(mongodb_database="test_db")
    assert get_database(test_app, s) is None

    mock_client = MagicMock()
    test_app.state.mongo_client = mock_client
    db = get_database(test_app, s)
    mock_client.__getitem__.assert_called_once_with("test_db")
    assert db == mock_client["test_db"]


def test_health_endpoint_preserved() -> None:
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}


def test_health_db_unconfigured() -> None:
    with patch.object(app_main.settings, "mongodb_uri", ""):
        with TestClient(app) as client:
            resp = client.get("/health/db")
            assert resp.status_code == 503
            assert resp.json() == {"status": "unavailable"}


def test_health_db_connected() -> None:
    mock_client = MagicMock()
    with patch.object(app_main.settings, "mongodb_uri", "mongodb://fake:27017"), \
         patch("app.main.create_mongo_client", return_value=mock_client), \
         patch("app.main.ping_mongo_database", return_value=True):
        with TestClient(app) as client:
            resp = client.get("/health/db")
            assert resp.status_code == 200
            assert resp.json() == {"status": "connected"}


def test_health_db_unavailable() -> None:
    mock_client = MagicMock()
    with patch.object(app_main.settings, "mongodb_uri", "mongodb://fake:27017"), \
         patch("app.main.create_mongo_client", return_value=mock_client), \
         patch("app.main.ping_mongo_database", return_value=False):
        with TestClient(app) as client:
            resp = client.get("/health/db")
            assert resp.status_code == 503
            assert resp.json() == {"status": "unavailable"}


def test_lifespan_closes_client_on_shutdown() -> None:
    mock_client = MagicMock()
    with patch.object(app_main.settings, "mongodb_uri", "mongodb://fake:27017"), \
         patch("app.main.create_mongo_client", return_value=mock_client), \
         patch("app.main.ping_mongo_database", return_value=True):
        with TestClient(app):
            pass
        mock_client.close.assert_called_once()


def test_degraded_startup_and_recovery() -> None:
    mock_client = MagicMock()
    # At startup ping fails (degraded), then later GET /health/db ping succeeds
    with patch.object(app_main.settings, "mongodb_uri", "mongodb://fake:27017"), \
         patch("app.main.create_mongo_client", return_value=mock_client), \
         patch("app.main.ping_mongo_database", side_effect=[False, True]):
        with TestClient(app) as client:
            assert client.app.state.mongo_client is mock_client
            resp = client.get("/health/db")
            assert resp.status_code == 200
            assert resp.json() == {"status": "connected"}


def test_no_secret_leak_in_logs_and_response(caplog: pytest.LogCaptureFixture) -> None:
    sensitive_uri = "mongodb+srv://secret_user:super_secret_pw@cluster0.example.mongodb.net/versionwise"
    mock_client = MagicMock()
    mock_client.admin.command.side_effect = ServerSelectionTimeoutError(
        f"Timeout connecting to cluster0.example.mongodb.net with uri {sensitive_uri}"
    )

    with caplog.at_level(logging.WARNING):
        result = ping_mongo_database(mock_client)
        assert result is False
        assert "secret_user" not in caplog.text
        assert "super_secret_pw" not in caplog.text
        assert "cluster0.example.mongodb.net" not in caplog.text
        assert "MongoDB ping failed: server selection or connection timeout." in caplog.text

