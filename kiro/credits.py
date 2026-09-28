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
# You should have received a copy of the GNU Affero General License
# along with this program. If not, see <https://www.gnu.org/licenses/>.

"""
Kiro Credits (usage limits) query.

Calls the Kiro control plane endpoint that the official IDE extension uses:

    GET https://management.<region>.kiro.dev/getUsageLimits?origin=AI_EDITOR&profileArn=<arn>
    Authorization: Bearer <accessToken>
    X-Kiro-Idp: <Google|Github|AWSIdC|BuilderId>
    X-Kiro-Profile-Arn: <arn>

The response contains usageBreakdownList[] (per-resource usage/limit/reset
dates, trial info, bonus credits) and subscriptionInfo.

Reverse-engineered from the Kiro IDE extension (kiro-agent), same as the
KiroTokenShare project.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Tuple

import httpx
from loguru import logger

from kiro.config import CREDITS_CACHE_TTL, KIRO_IDE_VERSION

# Kiro management endpoint template (control plane, not the runtime API)
MANAGEMENT_URL_TEMPLATE = "https://management.{region}.kiro.dev/getUsageLimits"


def provider_to_idp(provider: Optional[str]) -> str:
    """
    Map the token file's provider field to the X-Kiro-Idp header value.

    Reverse-engineered from the Kiro IDE extension (kiro-agent).

    Args:
        provider: Provider field from credentials ("Google", "Github",
                  "Enterprise", "ExternalIdp", "BuilderId", ...)

    Returns:
        Header value: Google / Github / AWSIdC / BuilderId
    """
    if provider == "Google":
        return "Google"
    if provider == "Github":
        return "Github"
    if provider in ("Enterprise", "ExternalIdp"):
        return "AWSIdC"
    return "BuilderId"


def region_from_profile_arn(arn: Optional[str]) -> str:
    """
    Extract region from "arn:aws:codewhisperer:REGION:..." ARN.

    Args:
        arn: Profile ARN string

    Returns:
        Region string, empty if the ARN doesn't match the expected shape
    """
    if not arn:
        return ""
    parts = arn.split(":")
    if len(parts) < 4 or parts[0] != "arn":
        return ""
    return parts[3]


class CreditsService:
    """
    Queries Kiro usage limits with a short TTL cache.

    The cache protects the Kiro management endpoint from being hammered by
    concurrent requests (e.g. external keys checking credits threshold on
    every request, plus admin page refreshes).

    Usage:
        service = CreditsService()
        data, error = await service.query(account, http_client)
    """

    def __init__(self, cache_ttl: int = CREDITS_CACHE_TTL):
        self._cache_ttl = max(0, cache_ttl)
        # account_id -> (timestamp, data or None, error or None)
        self._cache: Dict[str, Tuple[float, Optional[dict], Optional[str]]] = {}
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, account_id: str) -> asyncio.Lock:
        """One in-flight query per account (coalesces concurrent requests)."""
        if account_id not in self._locks:
            self._locks[account_id] = asyncio.Lock()
        return self._locks[account_id]

    async def query(
        self,
        account_id: str,
        auth_manager,
        http_client: Optional[httpx.AsyncClient],
        force_refresh: bool = False,
    ) -> Tuple[Optional[dict], Optional[str]]:
        """
        Query usage limits for one account.

        Args:
            account_id: Account identifier (cache key)
            auth_manager: KiroAuthManager with token/arn/provider
            http_client: Shared httpx client (a new one is created if omitted)
            force_refresh: Skip the cache

        Returns:
            (data, None) on success, (None, error_message) on failure
        """
        if not force_refresh and self._cache_ttl > 0:
            cached = self._cache.get(account_id)
            if cached and time.time() - cached[0] < self._cache_ttl:
                return cached[1], cached[2]

        lock = self._get_lock(account_id)
        async with lock:
            # Double-check cache after acquiring the lock (another request
            # may have refreshed it while we were waiting)
            if not force_refresh and self._cache_ttl > 0:
                cached = self._cache.get(account_id)
                if cached and time.time() - cached[0] < self._cache_ttl:
                    return cached[1], cached[2]

            data, error = await self._do_query(auth_manager, http_client)
            self._cache[account_id] = (time.time(), data, error)
            return data, error

    async def _do_query(
        self,
        auth_manager,
        http_client: Optional[httpx.AsyncClient],
    ) -> Tuple[Optional[dict], Optional[str]]:
        """Perform the actual HTTP request to the Kiro management endpoint."""
        try:
            token = await auth_manager.get_access_token()
        except Exception as e:
            return None, f"Failed to obtain access token: {e}"

        profile_arn = auth_manager.profile_arn or ""

        # Region priority: profile ARN > auth manager region > default
        region = region_from_profile_arn(profile_arn) or auth_manager.region or "us-east-1"

        params = {"origin": "AI_EDITOR"}
        if profile_arn:
            params["profileArn"] = profile_arn

        url = MANAGEMENT_URL_TEMPLATE.format(region=region)
        headers = {
            "Authorization": f"Bearer {token}",
            "X-Kiro-Idp": provider_to_idp(auth_manager.provider),
            "Accept": "application/json",
            "User-Agent": f"KiroIDE-{KIRO_IDE_VERSION}-{auth_manager.fingerprint}",
        }
        if profile_arn:
            headers["X-Kiro-Profile-Arn"] = profile_arn

        owns_client = http_client is None
        client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(timeout=30.0))
        try:
            try:
                response = await client.get(url, params=params, headers=headers)
            except httpx.HTTPError as e:
                return None, f"Request to Kiro management endpoint failed: {e}"

            if response.status_code != 200:
                body = response.text[:300]
                return None, f"Kiro returned {response.status_code}: {body}"

            try:
                data = response.json()
            except Exception as e:
                return None, f"Failed to parse Kiro response: {e}"
            return data, None
        finally:
            if owns_client:
                try:
                    await client.aclose()
                except Exception:
                    pass

    def get_cached(self, account_id: str) -> Optional[dict]:
        """Return cached data for an account without querying (or None)."""
        cached = self._cache.get(account_id)
        if cached and cached[1] is not None:
            return cached[1]
        return None

    def remaining_credits(self, data: Optional[dict]) -> Optional[float]:
        """
        Extract the main remaining credits value from a getUsageLimits response.

        Uses the first usageBreakdownList entry (the primary "Credits" resource)
        and computes limit - currentUsage. Returns None when the response has
        no usable breakdown (e.g. enterprise managed accounts).

        Args:
            data: Parsed getUsageLimits response

        Returns:
            Remaining credits as float, or None if not determinable
        """
        if not isinstance(data, dict):
            return None
        breakdowns = data.get("usageBreakdownList") or []
        for b in breakdowns:
            if not isinstance(b, dict):
                continue
            limit = b.get("usageLimitWithPrecision", b.get("usageLimit"))
            used = b.get("currentUsageWithPrecision", b.get("currentUsage"))
            if isinstance(limit, (int, float)) and isinstance(used, (int, float)):
                try:
                    return float(limit) - float(used)
                except (TypeError, ValueError):
                    continue
        return None


# Module-level shared instance (used by routes; created per-process)
credits_service = CreditsService()
