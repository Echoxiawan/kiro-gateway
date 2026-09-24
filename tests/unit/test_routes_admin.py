# -*- coding: utf-8 -*-

"""
Unit tests for admin console routes (routes_admin.py).

Covers:
- /admin page serving
- Login session lifecycle (correct/wrong password, cookie, logout)
- Key CRUD via /admin/keys endpoints
- /admin/overview summary
- /v1/credits auth (admin key, external key, invalid key) and rate limiting
"""

import pytest
from unittest.mock import MagicMock, AsyncMock, patch

import kiro.routes_admin as routes_admin
from kiro.routes_admin import _session_token_for, _SESSION_SALT
from kiro.api_key_manager import APIKeyManager


# =============================================================================
# /admin page
# =============================================================================

class TestAdminPage:
    def test_admin_page_served(self, test_client):
        response = test_client.get("/admin")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Kiro Gateway" in response.text


# =============================================================================
# Login / session
# =============================================================================

class TestAdminLogin:
    @pytest.fixture(autouse=True)
    def reset_session(self):
        # Reset the process-global session state around each test
        routes_admin._active_session = ""
        yield
        routes_admin._active_session = ""

    def _set_password(self, password):
        # Patch the module-level ADMIN_PASSWORD import
        return patch.object(routes_admin, "ADMIN_PASSWORD", password)

    def test_login_disabled_without_password(self, test_client):
        with self._set_password(""):
            response = test_client.post("/admin/login", json={"password": "x"})
            assert response.status_code == 403

    def test_login_wrong_password(self, test_client):
        with self._set_password("secret123"):
            response = test_client.post("/admin/login", json={"password": "nope"})
            assert response.status_code == 401

    def test_login_correct_password_sets_cookie(self, test_client):
        with self._set_password("secret123"):
            response = test_client.post("/admin/login", json={"password": "secret123"})
            assert response.status_code == 200
            assert response.json()["ok"] is True
            assert routes_admin.ADMIN_COOKIE_NAME in response.cookies

    def test_login_bad_json(self, test_client):
        with self._set_password("secret123"):
            response = test_client.post("/admin/login", content=b"not json")
            assert response.status_code == 400

    def test_protected_endpoint_requires_login(self, test_client):
        # 401 when console enabled but not logged in,
        # 403 when console disabled (no ADMIN_PASSWORD)
        response = test_client.get("/admin/keys")
        assert response.status_code in (401, 403)

    def test_session_token_derived_from_password(self):
        tok = _session_token_for("abc")
        assert tok == _session_token_for("abc")
        assert tok != _session_token_for("abd")

    def test_logout_clears_session(self, test_client):
        with self._set_password("secret123"):
            test_client.post("/admin/login", json={"password": "secret123"})
            response = test_client.post("/admin/logout")
            assert response.status_code == 200
            # After logout the protected endpoint rejects again
            r2 = test_client.get("/admin/keys")
            assert r2.status_code == 401


# =============================================================================
# Key CRUD
# =============================================================================

class TestKeysFlow:
    @pytest.fixture
    def client(self, test_client, tmp_path):
        routes_admin._active_session = ""
        with patch.object(routes_admin, "ADMIN_PASSWORD", "pw"):
            test_client.post("/admin/login", json={"password": "pw"})
            test_client.app.state.api_key_manager = APIKeyManager(
                keys_file=str(tmp_path / "keys.json"))
            yield test_client
        routes_admin._active_session = ""

    def test_create_key(self, client):
        r = client.post("/admin/keys", json={"name": "bob", "tokenQuota": 1000, "creditsThreshold": 5})
        assert r.status_code == 200
        body = r.json()
        assert body["ok"] is True
        assert body["key"]["name"] == "bob"
        assert body["key"]["tokenQuota"] == 1000
        assert body["key"]["key"].startswith("sk-gw-")

    def test_list_keys_returns_full_key(self, client):
        client.post("/admin/keys", json={"name": "bob"})
        r = client.get("/admin/keys")
        assert r.status_code == 200
        keys = r.json()["keys"]
        assert len(keys) == 1
        assert keys[0]["key"].startswith("sk-gw-")  # full, not masked

    def test_update_key(self, client):
        create = client.post("/admin/keys", json={"name": "bob"}).json()["key"]
        r = client.post("/admin/keys/update", json={"key": create["key"], "name": "bobby", "tokenQuota": 42})
        assert r.status_code == 200
        assert r.json()["key"]["name"] == "bobby"
        assert r.json()["key"]["tokenQuota"] == 42

    def test_update_missing_key_404(self, client):
        r = client.post("/admin/keys/update", json={"key": "sk-gw-missing", "name": "x"})
        assert r.status_code == 404

    def test_delete_key(self, client):
        create = client.post("/admin/keys", json={"name": "bob"}).json()["key"]
        r = client.post("/admin/keys/delete", json={"key": create["key"]})
        assert r.status_code == 200
        assert client.get("/admin/keys").json()["keys"] == []

    def test_delete_missing_key_404(self, client):
        r = client.post("/admin/keys/delete", json={"key": "sk-gw-missing"})
        assert r.status_code == 404

    def test_reset_usage(self, client):
        create = client.post("/admin/keys", json={"name": "bob"}).json()["key"]
        km = client.app.state.api_key_manager
        import asyncio
        asyncio.run(km.record_usage(create["key"], 100))
        r = client.post("/admin/keys/reset", json={"key": create["key"]})
        assert r.status_code == 200
        assert r.json()["key"]["tokensUsed"] == 0

    def test_key_manager_not_initialized_503(self, client):
        client.app.state.api_key_manager = None
        r = client.get("/admin/keys")
        assert r.status_code == 503


class TestOverview:
    def test_overview_returns_keys_summary(self, test_client, tmp_path):
        routes_admin._active_session = ""
        with patch.object(routes_admin, "ADMIN_PASSWORD", "pw"):
            test_client.post("/admin/login", json={"password": "pw"})
            test_client.app.state.api_key_manager = APIKeyManager(
                keys_file=str(tmp_path / "keys.json"))
            test_client.post("/admin/keys", json={"name": "zoe"})
            r = test_client.get("/admin/overview")
            assert r.status_code == 200
            body = r.json()
            assert len(body["keys"]) == 1
            assert "accounts" in body
            assert "accountSystem" in body
        routes_admin._active_session = ""


# =============================================================================
# /v1/credits auth and rate limiting
# =============================================================================

class TestCreditsEndpoint:
    @pytest.fixture(autouse=True)
    def reset_rate_log(self):
        routes_admin._credits_query_log.clear()
        yield
        routes_admin._credits_query_log.clear()

    def test_credits_requires_auth(self, test_client):
        r = test_client.get("/v1/credits")
        assert r.status_code == 401

    def test_credits_rejects_invalid_key(self, test_client):
        r = test_client.get("/v1/credits", headers={"Authorization": "Bearer wrong"})
        assert r.status_code == 401

    def test_credits_accepts_admin_key(self, test_client):
        from kiro.config import PROXY_API_KEY
        # account_manager is set by lifespan in test_client; credits query
        # will hit the (blocked) network and return 502, but auth must pass
        r = test_client.get("/v1/credits", headers={"Authorization": f"Bearer {PROXY_API_KEY}"})
        assert r.status_code != 401

    def test_credits_accepts_valid_external_key(self, test_client, tmp_path):
        from kiro.api_key_manager import APIKeyManager
        import asyncio
        km = APIKeyManager(keys_file=str(tmp_path / "keys.json"))
        rec = asyncio.run(km.create_key(name="ext"))
        test_client.app.state.api_key_manager = km
        routes_admin._credits_query_log.clear()
        r = test_client.get("/v1/credits", headers={"Authorization": f"Bearer {rec.key}"})
        assert r.status_code != 401

    def test_credits_rate_limits_external_key(self, test_client, tmp_path):
        from kiro.api_key_manager import APIKeyManager
        from kiro import routes_admin as ra
        import asyncio
        km = APIKeyManager(keys_file=str(tmp_path / "keys.json"))
        rec = asyncio.run(km.create_key(name="ext"))
        test_client.app.state.api_key_manager = km
        ra._credits_query_log.clear()

        with patch.object(ra, "CREDITS_QUERY_LIMIT", 2):
            statuses = []
            for _ in range(3):
                r = test_client.get("/v1/credits", headers={"Authorization": f"Bearer {rec.key}"})
                statuses.append(r.status_code)
            # The first CREDITS_QUERY_LIMIT queries pass auth (may 502 on network),
            # the third is rate limited with 429
            assert 429 in statuses
            assert statuses[-1] == 429

    def test_credits_no_rate_limit_for_admin_key(self, test_client):
        from kiro.config import PROXY_API_KEY
        from kiro import routes_admin as ra
        ra._credits_query_log.clear()
        with patch.object(ra, "CREDITS_QUERY_LIMIT", 1):
            statuses = [
                test_client.get("/v1/credits", headers={"Authorization": f"Bearer {PROXY_API_KEY}"}).status_code
                for _ in range(3)
            ]
        assert 429 not in statuses
