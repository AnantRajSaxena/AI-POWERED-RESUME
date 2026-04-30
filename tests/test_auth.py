import json
import pytest
from app import app, db, User


@pytest.fixture(autouse=True)
def setup_db():
    """Fresh in-memory DB for every test."""
    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///:memory:"
    app.config["TESTING"] = True
    with app.app_context():
        db.drop_all()
        db.create_all()
    yield
    with app.app_context():
        db.session.remove()
        db.drop_all()


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------

def test_register_and_login():
    client = app.test_client()
    r = client.post("/register", json={"username": "pytest_user", "password": "TestPass123"})
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

    r2 = client.post("/login", json={"username": "pytest_user", "password": "TestPass123"})
    assert r2.status_code == 200
    assert r2.get_json().get("user") == "pytest_user"


def test_register_duplicate():
    client = app.test_client()
    client.post("/register", json={"username": "dupe", "password": "StrongPass1"})
    r2 = client.post("/register", json={"username": "dupe", "password": "StrongPass1"})
    assert r2.status_code == 400
    assert r2.get_json().get("error") == "user_exists"


def test_login_wrong_password():
    client = app.test_client()
    client.post("/register", json={"username": "wpuser", "password": "rightpw"})
    r2 = client.post("/login", json={"username": "wpuser", "password": "wrongpw"})
    assert r2.status_code == 400
    assert r2.get_json().get("error") == "invalid_credentials"


def test_register_invalid_username():
    client = app.test_client()
    # too short
    r = client.post("/register", json={"username": "x", "password": "ValidPass1"})
    assert r.status_code == 400
    # contains space
    r2 = client.post("/register", json={"username": "bad name", "password": "ValidPass1"})
    assert r2.status_code == 400


def test_register_password_too_short():
    client = app.test_client()
    r = client.post("/register", json={"username": "shortpw", "password": "abc"})
    assert r.status_code == 400
    assert r.get_json().get("error") == "password_too_short"


def test_logout():
    client = app.test_client()
    client.post("/register", json={"username": "logoutuser", "password": "Pass1234"})
    client.post("/login", json={"username": "logoutuser", "password": "Pass1234"})
    r = client.get("/logout")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"


# ---------------------------------------------------------------------------
# Game route tests
# ---------------------------------------------------------------------------

def test_state_endpoint():
    client = app.test_client()
    r = client.get("/state")
    assert r.status_code == 200
    j = r.get_json()
    assert "state" in j
    assert "width" in j
    assert "height" in j


def test_step_endpoint():
    client = app.test_client()
    client.get("/reset")
    r = client.post("/step", json={"action": 3})  # move right
    assert r.status_code == 200
    j = r.get_json()
    assert "state" in j
    assert "reward" in j
    assert "done" in j


def test_reset_endpoint():
    client = app.test_client()
    r = client.get("/reset")
    assert r.status_code == 200
    assert "state" in r.get_json()


def test_help_endpoint():
    client = app.test_client()
    client.get("/reset")
    r = client.get("/help")
    assert r.status_code == 200
    j = r.get_json()
    assert "action" in j
    assert "source" in j


def test_level_locked():
    client = app.test_client()
    r = client.post("/level", json={"level": 99})
    assert r.status_code == 400
    assert r.get_json().get("error") == "level_locked"
