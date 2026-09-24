# -*- coding: utf-8 -*-

"""
Unit tests for APIKeyManager (api_key_manager.py).

Covers:
- Key generation format
- create / verify / record_usage / update / delete lifecycle
- Expiration and quota enforcement in verify_sync
- Save/load roundtrip and corrupt-file fallback
"""

import json
import time
from pathlib import Path

import pytest

from kiro.api_key_manager import APIKeyManager, APIKeyRecord


@pytest.fixture
def keys_file(tmp_path):
    return str(tmp_path / "api_keys.json")


@pytest.fixture
def manager(keys_file):
    return APIKeyManager(keys_file=keys_file)


def _get(manager, key):
    return manager._records.get(key)


class TestGenerateKey:
    def test_key_format_has_prefix(self, manager):
        key = manager.generate_key()
        assert key.startswith("sk-gw-")
        body = key[len("sk-gw-"):]
        assert len(body) == 32
        int(body, 16)  # hex

    def test_keys_are_unique(self, manager):
        keys = {manager.generate_key() for _ in range(50)}
        assert len(keys) == 50


class TestCreateAndVerify:
    @pytest.mark.asyncio
    async def test_create_and_verify(self, manager):
        rec = await manager.create_key(name="test", token_quota=1000, credits_threshold=0)
        assert rec.key.startswith("sk-gw-")
        assert rec.enabled is True
        assert rec.tokenQuota == 1000
        assert rec.tokensUsed == 0

        record = manager.verify_sync(rec.key)
        assert record is not None
        assert record.name == "test"

    @pytest.mark.asyncio
    async def test_verify_unknown_key_returns_none(self, manager):
        assert manager.verify_sync("sk-gw-nonexistent") is None
        assert manager.verify_sync("") is None

    @pytest.mark.asyncio
    async def test_verify_disabled_key_returns_none(self, manager):
        rec = await manager.create_key(name="x")
        await manager.update_key(rec.key, {"enabled": False})
        assert manager.verify_sync(rec.key) is None

    @pytest.mark.asyncio
    async def test_verify_expired_key_returns_none(self, manager):
        rec = await manager.create_key(name="x", expires_at=int(time.time()) - 10)
        assert manager.verify_sync(rec.key) is None

    @pytest.mark.asyncio
    async def test_verify_future_expiry_key_works(self, manager):
        rec = await manager.create_key(name="x", expires_at=int(time.time()) + 3600)
        assert manager.verify_sync(rec.key) is not None

    @pytest.mark.asyncio
    async def test_verify_key_over_quota_returns_none(self, manager):
        rec = await manager.create_key(name="x", token_quota=100)
        await manager.record_usage(rec.key, 150)
        assert manager.verify_sync(rec.key) is None

    @pytest.mark.asyncio
    async def test_verify_key_at_quota_edge_still_passes(self, manager):
        # used < quota passes; used >= quota fails
        rec = await manager.create_key(name="x", token_quota=100)
        await manager.record_usage(rec.key, 99)
        assert manager.verify_sync(rec.key) is not None

    @pytest.mark.asyncio
    async def test_zero_quota_means_unlimited(self, manager):
        rec = await manager.create_key(name="x", token_quota=0)
        await manager.record_usage(rec.key, 10_000_000)
        assert manager.verify_sync(rec.key) is not None

    @pytest.mark.asyncio
    async def test_create_duplicate_key_raises(self, manager):
        rec = await manager.create_key(name="x")
        with pytest.raises(ValueError):
            await manager.create_key(name="y", key=rec.key)


class TestRecordUsage:
    @pytest.mark.asyncio
    async def test_record_usage_accumulates(self, manager):
        rec = await manager.create_key(name="x")
        await manager.record_usage(rec.key, 100)
        await manager.record_usage(rec.key, 50)
        r = _get(manager, rec.key)
        assert r.tokensUsed == 150
        assert r.requestsUsed == 2

    @pytest.mark.asyncio
    async def test_record_usage_unknown_key_ignored(self, manager):
        await manager.record_usage("sk-gw-unknown", 100)  # no raise
        assert await manager.list_keys() == []

    @pytest.mark.asyncio
    async def test_create_persists_to_file(self, manager, keys_file):
        rec = await manager.create_key(name="x", token_quota=42)
        await manager.record_usage(rec.key, 10)
        await manager.flush()
        assert Path(keys_file).exists()
        data = json.loads(open(keys_file).read())
        assert data[0]["tokensUsed"] == 10
        assert data[0]["tokenQuota"] == 42


class TestSaveLoadRoundtrip:
    @pytest.mark.asyncio
    async def test_roundtrip_preserves_records(self, keys_file):
        m1 = APIKeyManager(keys_file=keys_file)
        rec = await m1.create_key(name="alice", token_quota=500, credits_threshold=8)
        await m1.record_usage(rec.key, 123)
        await m1.flush()

        m2 = APIKeyManager(keys_file=keys_file)
        await m2.load()
        loaded = _get(m2, rec.key)
        assert loaded is not None
        assert loaded.name == "alice"
        assert loaded.tokenQuota == 500
        assert loaded.creditsThreshold == 8
        assert loaded.tokensUsed == 123
        assert loaded.requestsUsed == 1

    @pytest.mark.asyncio
    async def test_load_missing_file_starts_empty(self, keys_file):
        m = APIKeyManager(keys_file=keys_file)
        await m.load()
        assert await m.list_keys() == []

    @pytest.mark.asyncio
    async def test_load_corrupt_file_falls_back_to_empty(self, keys_file):
        with open(keys_file, "w") as f:
            f.write("not json at all {{{")
        m = APIKeyManager(keys_file=keys_file)
        await m.load()  # should not raise
        assert await m.list_keys() == []


class TestUpdateDeleteReset:
    @pytest.mark.asyncio
    async def test_update_changes_fields(self, manager):
        rec = await manager.create_key(name="old", token_quota=100)
        updated = await manager.update_key(rec.key, {"name": "new", "tokenQuota": 200})
        assert updated.name == "new"
        assert updated.tokenQuota == 200

    @pytest.mark.asyncio
    async def test_update_unknown_key_raises(self, manager):
        with pytest.raises(KeyError):
            await manager.update_key("sk-gw-nope", {"name": "x"})

    @pytest.mark.asyncio
    async def test_delete_removes_key(self, manager):
        rec = await manager.create_key(name="x")
        assert await manager.delete_key(rec.key) is True
        assert manager.verify_sync(rec.key) is None
        assert await manager.delete_key(rec.key) is False

    @pytest.mark.asyncio
    async def test_reset_usage_zeroes_counters(self, manager):
        rec = await manager.create_key(name="x", token_quota=100)
        await manager.record_usage(rec.key, 90)
        await manager.reset_usage(rec.key)
        r = _get(manager, rec.key)
        assert r.tokensUsed == 0
        assert r.requestsUsed == 0

    @pytest.mark.asyncio
    async def test_reset_unknown_key_raises(self, manager):
        with pytest.raises(KeyError):
            await manager.reset_usage("sk-gw-nope")


class TestPublicDict:
    def test_mask_key(self):
        rec = APIKeyRecord(key="sk-gw-" + "a" * 32, name="x")
        masked = rec.masked_key()
        full = rec.key
        assert masked != full
        # prefix + short suffix visible, middle hidden
        assert masked.startswith("sk-gw-")
        assert masked == f"{full[:8]}...{full[-4:]}"

    def test_to_public_dict_hides_full_key(self):
        key = "sk-gw-" + "abcd" * 8
        rec = APIKeyRecord(key=key, name="x")
        d = rec.to_public_dict(mask_key=True)
        assert d["key"] != key
        d2 = rec.to_public_dict(mask_key=False)
        assert d2["key"] == key
