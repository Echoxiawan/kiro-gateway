# -*- coding: utf-8 -*-

"""
Unit tests for the credits query service (credits.py).

Covers:
- provider -> X-Kiro-Idp header mapping
- region extraction from profile ARN
- URL construction and request headers
- remaining_credits parsing
- TTL cache behavior
"""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from kiro.credits import (
    CreditsService,
    MANAGEMENT_URL_TEMPLATE,
    provider_to_idp,
    region_from_profile_arn,
    credits_service,
)


# =============================================================================
# Header / URL helpers
# =============================================================================

class TestProviderToIdp:
    def test_google(self):
        assert provider_to_idp("Google") == "Google"

    def test_github(self):
        assert provider_to_idp("Github") == "Github"

    def test_enterprise_maps_to_awsidc(self):
        assert provider_to_idp("Enterprise") == "AWSIdC"

    def test_externalidp_maps_to_awsidc(self):
        assert provider_to_idp("ExternalIdp") == "AWSIdC"

    def test_builderid_default(self):
        assert provider_to_idp("BuilderId") == "BuilderId"

    def test_none_defaults_to_builderid(self):
        assert provider_to_idp(None) == "BuilderId"

    def test_unknown_defaults_to_builderid(self):
        assert provider_to_idp("SomethingElse") == "BuilderId"

    def test_case_sensitive(self):
        # "google" (lowercase) is not a recognized provider
        assert provider_to_idp("google") == "BuilderId"


class TestRegionFromArn:
    def test_extracts_region(self):
        arn = "arn:aws:codewhisperer:ap-southeast-2:123456789012:profile/ABC"
        assert region_from_profile_arn(arn) == "ap-southeast-2"

    def test_us_east_1(self):
        arn = "arn:aws:codewhisperer:us-east-1:999:profile/XYZ"
        assert region_from_profile_arn(arn) == "us-east-1"

    def test_none_returns_empty(self):
        assert region_from_profile_arn(None) == ""

    def test_empty_returns_empty(self):
        assert region_from_profile_arn("") == ""

    def test_malformed_returns_empty(self):
        assert region_from_profile_arn("not:an:arn") == ""


def _make_auth_manager(provider="Enterprise", region="ap-southeast-2",
                       profile_arn="arn:aws:codewhisperer:ap-southeast-2:111:profile/T1", token="tok123"):
    am = MagicMock()
    am.provider = provider
    am.region = region
    am.profile_arn = profile_arn
    am.fingerprint = "FPFPFPFP"
    am.get_access_token = AsyncMock(return_value=token)
    return am


def _make_client(response_json=None, status_code=200, raise_error=None):
    resp = MagicMock()
    resp.status_code = status_code
    resp.text = ""
    resp.json = MagicMock(return_value=response_json or {})
    client = MagicMock()
    client.aclose = AsyncMock()
    if raise_error:
        client.get = AsyncMock(side_effect=raise_error)
    else:
        client.get = AsyncMock(return_value=resp)
    return client, client.get


# =============================================================================
# Query behavior
# =============================================================================

class TestQuery:
    @pytest.mark.asyncio
    async def test_url_and_headers(self):
        am = _make_auth_manager()
        client, get = _make_client({"ok": True})

        service = CreditsService(cache_ttl=0)
        data, err = await service.query("acct1", am, client)

        assert err is None
        assert data == {"ok": True}
        # Called once (cache disabled)
        assert get.await_count == 1

        url = get.await_args.args[0] if get.await_args.args else get.await_args.kwargs.get("url")
        kwargs = get.await_args.kwargs
        assert url == "https://management.ap-southeast-2.kiro.dev/getUsageLimits"
        assert kwargs["params"]["origin"] == "AI_EDITOR"
        assert kwargs["params"]["profileArn"].endswith("profile/T1")
        headers = kwargs["headers"]
        assert headers["Authorization"] == "Bearer tok123"
        assert headers["X-Kiro-Idp"] == "AWSIdC"
        assert headers["X-Kiro-Profile-Arn"] == am.profile_arn
        assert headers["User-Agent"].startswith("KiroIDE-1.1.14-")

    @pytest.mark.asyncio
    async def test_non_200_returns_error(self):
        am = _make_auth_manager()
        client, get = _make_client({"message": "denied"}, status_code=403)

        service = CreditsService(cache_ttl=0)
        data, err = await service.query("acct1", am, client)

        assert data is None
        assert "403" in err

    @pytest.mark.asyncio
    async def test_http_error_returns_error(self):
        am = _make_auth_manager()
        client, get = _make_client(raise_error=httpx.ConnectError("boom"))

        service = CreditsService(cache_ttl=0)
        data, err = await service.query("acct1", am, client)

        assert data is None
        assert err is not None

    @pytest.mark.asyncio
    async def test_token_failure_returns_error(self):
        am = _make_auth_manager()
        am.get_access_token = AsyncMock(side_effect=RuntimeError("no token"))
        client, get = _make_client()

        service = CreditsService(cache_ttl=0)
        data, err = await service.query("acct1", am, client)

        assert data is None
        assert "access token" in err.lower()

    @pytest.mark.asyncio
    async def test_region_falls_back_to_auth_manager_region(self):
        # ARN without valid region shape -> use auth_manager.region
        am = _make_auth_manager(profile_arn="weird-arn")
        client, get = _make_client()

        service = CreditsService(cache_ttl=0)
        await service.query("acct1", am, client)

        url = get.await_args.args[0]
        assert "ap-southeast-2" in url

    @pytest.mark.asyncio
    async def test_missing_profile_arn_omitted_from_headers(self):
        am = _make_auth_manager(profile_arn=None)
        client, get = _make_client()

        service = CreditsService(cache_ttl=0)
        await service.query("acct1", am, client)

        kwargs = get.await_args.kwargs
        assert "profileArn" not in kwargs["params"]
        assert "X-Kiro-Profile-Arn" not in kwargs["headers"]


class TestCache:
    @pytest.mark.asyncio
    async def test_ttl_cache_hits_once(self):
        am = _make_auth_manager()
        client, get = _make_client({"n": 1})

        service = CreditsService(cache_ttl=60)
        d1, _ = await service.query("acct", am, client)
        d2, _ = await service.query("acct", am, client)
        assert d1 == d2
        assert get.await_count == 1

    @pytest.mark.asyncio
    async def test_force_refresh_bypasses_cache(self):
        am = _make_auth_manager()
        client, get = _make_client({"n": 1})

        service = CreditsService(cache_ttl=60)
        await service.query("acct", am, client)
        await service.query("acct", am, client, force_refresh=True)
        assert get.await_count == 2

    @pytest.mark.asyncio
    async def test_zero_ttl_disables_cache(self):
        am = _make_auth_manager()
        client, get = _make_client({"n": 1})

        service = CreditsService(cache_ttl=0)
        await service.query("acct", am, client)
        await service.query("acct", am, client)
        assert get.await_count == 2

    @pytest.mark.asyncio
    async def test_errors_are_cached_too(self):
        am = _make_auth_manager()
        client, get = _make_client(None, status_code=500)

        service = CreditsService(cache_ttl=60)
        _, err1 = await service.query("acct", am, client)
        _, err2 = await service.query("acct", am, client)
        assert err1 is not None
        assert err1 == err2
        assert get.await_count == 1

    def test_get_cached(self):
        service = CreditsService(cache_ttl=60)
        service._cache["a"] = (time.time(), {"x": 1}, None)
        service._cache["b"] = (time.time(), None, "err")
        assert service.get_cached("a") == {"x": 1}
        assert service.get_cached("b") is None
        assert service.get_cached("missing") is None


class TestRemainingCredits:
    def test_simple(self):
        data = {"usageBreakdownList": [
            {"usageLimit": 100, "currentUsage": 30}
        ]}
        assert credits_service.remaining_credits(data) == 70

    def test_precision_fields_preferred(self):
        data = {"usageBreakdownList": [
            {"usageLimit": 100, "currentUsage": 30,
             "usageLimitWithPrecision": 100.5, "currentUsageWithPrecision": 0.5}
        ]}
        assert credits_service.remaining_credits(data) == 100

    def test_empty_breakdown(self):
        assert credits_service.remaining_credits({"usageBreakdownList": []}) is None

    def test_none_data(self):
        assert credits_service.remaining_credits(None) is None

    def test_non_dict_data(self):
        assert credits_service.remaining_credits("nope") is None

    def test_non_numeric_fields_skipped(self):
        data = {"usageBreakdownList": [
            {"usageLimit": "many", "currentUsage": "few"}
        ]}
        assert credits_service.remaining_credits(data) is None
