# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/Echoxiawan/kiro-gateway
# Copyright (C) 2026 Echoxiawan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Admin console and Credits routes for Kiro Gateway.

Endpoints:
- GET  /v1/credits            - query Kiro usage limits (any valid API key)
- GET  /admin                 - web management page (static HTML)
- POST /admin/login           - admin password login (cookie session)
- POST /admin/logout          - logout
- GET  /admin/keys            - list external keys (usage + quota)
- POST /admin/keys            - create key
- POST /admin/keys/update     - update key (name/enabled/quotas/expiration)
- POST /admin/keys/delete     - delete key
- POST /admin/keys/reset      - reset usage counters
- GET  /admin/credits         - query credits for all initialized accounts
- GET  /admin/overview        - accounts + stats + credits summary
"""

import asyncio
import hashlib
import secrets
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse
from loguru import logger

from kiro.config import (
    PROXY_API_KEY,
    ADMIN_PASSWORD,
    CREDITS_QUERY_LIMIT,
)
from kiro.routes_openai import verify_api_key, extract_bearer_key
from kiro.credits import credits_service

try:
    from kiro.debug_logger import debug_logger
except ImportError:
    debug_logger = None


ADMIN_COOKIE_NAME = "kgw_admin"

# Process-level session salt: cookie holds sha256(password + salt), restart invalidates
_SESSION_SALT = secrets.token_hex(16)
_active_session = ""

# /v1/credits rate limiting: key -> list of timestamps
_credits_query_log: dict = {}


def _session_token_for(password: str) -> str:
    """Compute the session cookie value for an admin password."""
    return hashlib.sha256((password + _SESSION_SALT).encode()).hexdigest()


def _is_valid_session(token: Optional[str]) -> bool:
    """Check the session cookie against the active session."""
    return bool(_active_session) and token == _active_session


router = APIRouter(tags=["Admin Console"])


# ==================================================================================================
# Credits endpoint (available to any valid key)
# ==================================================================================================

@router.get("/v1/credits", dependencies=[])
async def get_credits(request: Request):
    """
    Query Kiro account credits / usage limits.

    Available to any valid API key (admin PROXY_API_KEY or external keys).
    External keys are rate limited (CREDITS_QUERY_LIMIT per 60s window).

    Returns the raw getUsageLimits response from the Kiro management API:
    subscriptionInfo, usageBreakdownList[], nextDateReset, etc.
    """
    # Reuse the standard key verification (raises 401 on failure)
    key = extract_bearer_key(request.headers.get("authorization", ""))
    await verify_api_key(request, key or None)

    # Rate limit /v1/credits for non-admin keys
    key = extract_bearer_key(request.headers.get("authorization", ""))
    is_admin = bool(key) and key == PROXY_API_KEY
    if not is_admin and CREDITS_QUERY_LIMIT > 0:
        now = time.time()
        recent = [t for t in _credits_query_log.get(key, []) if now - t < 60]
        if len(recent) >= CREDITS_QUERY_LIMIT:
            raise HTTPException(
                status_code=429,
                detail="Credits query rate limit exceeded. Try again in a minute."
            )
        recent.append(now)
        _credits_query_log[key] = recent

    account_manager = getattr(request.app.state, "account_manager", None)
    if account_manager is None:
        raise HTTPException(status_code=503, detail="Account system not initialized")

    try:
        account = account_manager.get_first_account()
    except RuntimeError:
        raise HTTPException(status_code=503, detail="No initialized account available")

    data, error = await credits_service.query(
        account.id, account.auth_manager, request.app.state.http_client
    )
    if error:
        raise HTTPException(status_code=502, detail=error)

    return data


# ==================================================================================================
# Admin session management
# ==================================================================================================

async def require_admin(request: Request, kgw_admin: Optional[str] = Cookie(None)) -> None:
    """
    Dependency: verify the admin session cookie.

    Raises 401 when the admin console is disabled or the session is invalid.
    """
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Admin console disabled (ADMIN_PASSWORD not set)")
    if not _is_valid_session(kgw_admin):
        raise HTTPException(status_code=401, detail="Not logged in or session expired")


@router.post("/admin/login")
async def admin_login(request: Request, response: Response):
    """
    Log in to the admin console.

    Body: {"password": "..."}

    Sets an HttpOnly session cookie valid until process restart.
    """
    if not ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Admin console disabled (ADMIN_PASSWORD not set)")

    try:
        import json
        body = json.loads(await request.body() or b"{}")
    except Exception:
        raise HTTPException(status_code=400, detail="Request body is not valid JSON")

    password = body.get("password", "")
    if not password or password != ADMIN_PASSWORD:
        # Slow down brute force attempts
        await asyncio.sleep(0.3)
        raise HTTPException(status_code=401, detail="Incorrect admin password")

    global _active_session
    _active_session = _session_token_for(ADMIN_PASSWORD)
    response.set_cookie(
        ADMIN_COOKIE_NAME,
        _active_session,
        httponly=True,
        samesite="lax",
        path="/",
    )
    return {"ok": True}


@router.post("/admin/logout")
async def admin_logout(response: Response):
    """Log out from the admin console."""
    global _active_session
    _active_session = ""
    response.delete_cookie(ADMIN_COOKIE_NAME, path="/")
    return {"ok": True}


# ==================================================================================================
# Key management
# ==================================================================================================

def _get_keys_manager(request: Request):
    """Get APIKeyManager or raise 503."""
    manager = getattr(request.app.state, "api_key_manager", None)
    if manager is None:
        raise HTTPException(status_code=503, detail="API key manager not initialized")
    return manager


@router.get("/admin/keys", dependencies=[Depends(require_admin)])
async def admin_list_keys(request: Request):
    """List all external keys with quota/usage info. Keys are returned in full (admin needs to copy them)."""
    manager = _get_keys_manager(request)
    records = await manager.list_keys()
    return {"keys": [r.to_public_dict() for r in records]}


@router.post("/admin/keys", dependencies=[Depends(require_admin)])
async def admin_create_key(request: Request):
    """
    Create a new external API key.

    Body: {"name": "...", "tokenQuota": 5000000, "creditsThreshold": 0, "expiresAt": 0}
    """
    manager = _get_keys_manager(request)

    try:
        import json
        body = json.loads(await request.body() or b"{}")
    except Exception:
        raise HTTPException(status_code=400, detail="Request body is not valid JSON")

    try:
        record = await manager.create_key(
            name=str(body.get("name", "")),
            token_quota=int(body.get("tokenQuota", 0)),
            credits_threshold=int(body.get("creditsThreshold", 0)),
            expires_at=int(body.get("expiresAt", 0)),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"ok": True, "key": record.to_public_dict()}


@router.post("/admin/keys/update", dependencies=[Depends(require_admin)])
async def admin_update_key(request: Request):
    """
    Update an existing key.

    Body: {"key": "sk-gw-...", ...fields}
    Supported fields: name, enabled, tokenQuota, creditsThreshold, expiresAt
    """
    manager = _get_keys_manager(request)

    try:
        import json
        body = json.loads(await request.body() or b"{}")
    except Exception:
        raise HTTPException(status_code=400, detail="Request body is not valid JSON")

    key = body.get("key", "")
    if not key:
        raise HTTPException(status_code=400, detail="Missing 'key' field")

    updates = {k: v for k, v in body.items() if k != "key"}
    try:
        record = await manager.update_key(key, updates)
    except KeyError:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"ok": True, "key": record.to_public_dict()}


@router.post("/admin/keys/delete", dependencies=[Depends(require_admin)])
async def admin_delete_key(request: Request):
    """Delete a key. Body: {"key": "sk-gw-..."}"""
    manager = _get_keys_manager(request)

    try:
        import json
        body = json.loads(await request.body() or b"{}")
    except Exception:
        raise HTTPException(status_code=400, detail="Request body is not valid JSON")

    key = body.get("key", "")
    if not key:
        raise HTTPException(status_code=400, detail="Missing 'key' field")

    deleted = await manager.delete_key(key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"ok": True}


@router.post("/admin/keys/reset", dependencies=[Depends(require_admin)])
async def admin_reset_key(request: Request):
    """Reset usage counters for a key. Body: {"key": "sk-gw-..."}"""
    manager = _get_keys_manager(request)

    try:
        import json
        body = json.loads(await request.body() or b"{}")
    except Exception:
        raise HTTPException(status_code=400, detail="Request body is not valid JSON")

    key = body.get("key", "")
    if not key:
        raise HTTPException(status_code=400, detail="Missing 'key' field")

    try:
        record = await manager.reset_usage(key)
    except KeyError:
        raise HTTPException(status_code=404, detail="Key not found")
    return {"ok": True, "key": record.to_public_dict()}


# ==================================================================================================
# Credits & overview
# ==================================================================================================

@router.get("/admin/credits", dependencies=[Depends(require_admin)])
async def admin_credits(request: Request):
    """Query Kiro credits for all initialized accounts (each queried live with short cache)."""

    account_manager = getattr(request.app.state, "account_manager", None)
    if account_manager is None:
        raise HTTPException(status_code=503, detail="Account system not initialized")

    http_client = request.app.state.http_client
    results = []
    for account_id, account in account_manager._accounts.items():
        if account.auth_manager is None:
            continue
        data, error = await credits_service.query(account_id, account.auth_manager, http_client)
        entry = {
            "accountId": account_id,
            "ok": error is None,
        }
        if error:
            entry["error"] = error
        else:
            entry["data"] = data
        results.append(entry)

    return {"accounts": results}


@router.get("/admin/overview", dependencies=[Depends(require_admin)])
async def admin_overview(request: Request):
    """Summary: accounts with stats, external keys usage, app info."""

    account_manager = getattr(request.app.state, "account_manager", None)
    key_manager = _get_keys_manager(request)

    accounts = []
    if account_manager is not None:
        for account_id, account in account_manager._accounts.items():
            accounts.append({
                "id": account_id,
                "initialized": account.auth_manager is not None,
                "failures": account.failures,
                "stats": {
                    "total_requests": account.stats.total_requests,
                    "successful_requests": account.stats.successful_requests,
                    "failed_requests": account.stats.failed_requests,
                },
            })

    keys = [r.to_public_dict() for r in await key_manager.list_keys()]

    return {
        "accounts": accounts,
        "keys": keys,
        "accountSystem": getattr(request.app.state, "account_system", False),
    }


# ==================================================================================================
# Server address helper
# ==================================================================================================

@router.get("/admin/server-addr", dependencies=[Depends(require_admin)])
async def admin_server_addr(request: Request):
    """Return the server's actual LAN IP and port so the admin page can build a copyable Base URL."""
    import socket
    port = request.url.port or 8001
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = request.url.hostname or "127.0.0.1"
    return {"baseUrl": f"http://{ip}:{port}/v1", "ip": ip, "port": port}


# ==================================================================================================
# Static admin page
# ==================================================================================================

_ADMIN_PAGE_PATH = Path(__file__).parent / "static" / "admin.html"


@router.get("/admin")
async def admin_page():
    """Serve the admin console web page."""
    if not _ADMIN_PAGE_PATH.exists():
        raise HTTPException(status_code=500, detail="admin.html missing")
    return FileResponse(_ADMIN_PAGE_PATH, media_type="text/html")
