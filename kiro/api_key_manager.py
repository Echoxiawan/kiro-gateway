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
API Key Manager for Kiro Gateway.

Manages external API keys with quotas, separate from the admin PROXY_API_KEY.

Features:
- Multiple keys stored in api_keys.json (atomic writes)
- Per-key total token quota and expiration
- Per-key Kiro credits threshold (reject requests when account credits run low)
- Usage tracking (tokens used, requests used) with periodic persistence
- Thread-safe via asyncio.Lock

The admin key (PROXY_API_KEY from .env) always bypasses this manager:
it is unlimited and not tracked.
"""

import asyncio
import secrets
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from kiro.config import STATE_SAVE_INTERVAL_SECONDS

KEY_PREFIX = "sk-gw-"


@dataclass
class APIKeyRecord:
    """
    A single external API key with quota configuration.

    Attributes:
        key: The actual key string ("sk-gw-<hex>")
        name: Human-readable label shown in the admin UI
        enabled: Disabled keys are rejected immediately
        expiresAt: Unix seconds, 0 = never expires
        tokenQuota: Total token budget, 0 = unlimited
        creditsThreshold: Reject when account remaining credits < threshold, 0 = disabled
        tokensUsed: Cumulative tokens consumed through this key
        requestsUsed: Cumulative request count (statistics only)
        createdAt: Unix seconds
    """
    key: str
    name: str = ""
    enabled: bool = True
    expiresAt: int = 0
    tokenQuota: int = 0
    creditsThreshold: int = 0
    tokensUsed: int = 0
    requestsUsed: int = 0
    createdAt: int = field(default_factory=lambda: int(time.time()))

    def to_public_dict(self, mask_key: bool = False) -> Dict[str, Any]:
        """Serialize for API responses. mask_key replaces middle of key with asterisks."""
        d = asdict(self)
        if mask_key:
            d["key"] = self.masked_key()
        return d

    def masked_key(self) -> str:
        if len(self.key) <= 10:
            return "*" * len(self.key)
        return f"{self.key[:8]}...{self.key[-4:]}"


class APIKeyManager:
    """
    Manages external API keys.

    Responsibilities:
    - Load/save api_keys.json (atomic tmp+rename, same pattern as state.json)
    - Verify incoming keys (enabled / expiration / token quota)
    - Track usage and persist it periodically (dirty flag + background saver)

    Example:
        >>> manager = APIKeyManager("api_keys.json")
        >>> await manager.load()
        >>> record = await manager.verify("sk-gw-abc")
        >>> await manager.record_usage("sk-gw-abc", tokens=1234)
    """

    def __init__(self, keys_file: str):
        self._keys_file = keys_file
        self._records: Dict[str, APIKeyRecord] = {}
        self._lock = asyncio.Lock()
        self._dirty = False

    async def load(self) -> None:
        """Load keys from JSON file. Missing file = empty key set."""
        path = Path(self._keys_file).expanduser()
        if not path.exists():
            logger.debug(f"API keys file not found: {self._keys_file} (starting with no external keys)")
            return

        try:
            import json
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for entry in data if isinstance(data, list) else []:
                if not isinstance(entry, dict) or not entry.get("key"):
                    logger.warning(f"Invalid API key entry skipped: {entry}")
                    continue
                record = APIKeyRecord(
                    key=entry["key"],
                    name=entry.get("name", ""),
                    enabled=entry.get("enabled", True),
                    expiresAt=int(entry.get("expiresAt", 0)),
                    tokenQuota=int(entry.get("tokenQuota", 0)),
                    creditsThreshold=int(entry.get("creditsThreshold", 0)),
                    tokensUsed=int(entry.get("tokensUsed", 0)),
                    requestsUsed=int(entry.get("requestsUsed", 0)),
                    createdAt=int(entry.get("createdAt", int(time.time()))),
                )
                self._records[record.key] = record

            logger.info(f"Loaded {len(self._records)} external API key(s) from {self._keys_file}")
        except Exception as e:
            logger.error(f"Failed to load API keys: {e}")

    async def _save(self) -> None:
        """Save keys to JSON file atomically (tmp + rename)."""
        import json
        path = Path(self._keys_file).expanduser()
        tmp_path = path.with_suffix('.json.tmp')

        try:
            data = [asdict(r) for r in self._records.values()]
            with open(tmp_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            tmp_path.replace(path)
            logger.debug("API keys saved successfully")
        except Exception as e:
            logger.error(f"Failed to save API keys: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    async def save_state_periodically(self) -> None:
        """Background task: persist usage counters periodically when dirty."""
        while True:
            await asyncio.sleep(STATE_SAVE_INTERVAL_SECONDS)
            if self._dirty:
                async with self._lock:
                    await self._save()
                    self._dirty = False

    async def flush(self) -> None:
        """Force immediate persistence (used at shutdown)."""
        async with self._lock:
            await self._save()
            self._dirty = False

    def verify_sync(self, key: str) -> Optional[APIKeyRecord]:
        """
        Check a key without taking the async lock.

        Reads only immutable-ish fields (enabled/expiresAt/tokenQuota);
        usage counters are updated under lock elsewhere, so a stale read
        here only means we might allow one request past quota by a few tokens.

        Args:
            key: Raw key string from the Authorization header

        Returns:
            APIKeyRecord if the key is valid and within quota, None otherwise
        """
        record = self._records.get(key)
        if not record or not record.enabled:
            return None
        if record.expiresAt > 0 and time.time() > record.expiresAt:
            return None
        if record.tokenQuota > 0 and record.tokensUsed >= record.tokenQuota:
            return None
        return record

    async def record_usage(self, key: str, tokens: int) -> None:
        """
        Accumulate usage for a key after a request completes.

        Args:
            key: Key string
            tokens: Total tokens consumed (0 still counts as a request)
        """
        if key not in self._records:
            return
        async with self._lock:
            record = self._records[key]
            record.requestsUsed += 1
            record.tokensUsed += max(0, int(tokens))
            self._dirty = True

    def generate_key(self) -> str:
        """Generate a new random key string."""
        return KEY_PREFIX + secrets.token_hex(16)

    async def list_keys(self) -> List[APIKeyRecord]:
        """Return all key records."""
        async with self._lock:
            return list(self._records.values())

    async def create_key(
        self,
        name: str = "",
        token_quota: int = 0,
        credits_threshold: int = 0,
        expires_at: int = 0,
        key: Optional[str] = None,
    ) -> APIKeyRecord:
        """
        Create a new API key and persist it.

        Args:
            name: Human-readable label
            token_quota: Total token budget (0 = unlimited)
            credits_threshold: Kiro credits threshold (0 = disabled)
            expires_at: Unix seconds (0 = never)
            key: Explicit key string (auto-generated if omitted)

        Returns:
            The created record
        """
        async with self._lock:
            key = key or self.generate_key()
            if key in self._records:
                raise ValueError(f"Key already exists: {key[:8]}...")
            record = APIKeyRecord(
                key=key,
                name=name,
                tokenQuota=max(0, int(token_quota)),
                creditsThreshold=max(0, int(credits_threshold)),
                expiresAt=max(0, int(expires_at)),
            )
            self._records[key] = record
            await self._save()
            logger.info(f"Created API key '{name}' (quota={record.tokenQuota}, threshold={record.creditsThreshold})")
            return record

    async def update_key(self, key: str, updates: Dict[str, Any]) -> APIKeyRecord:
        """
        Update fields of an existing key.

        Supported fields: name, enabled, tokenQuota, creditsThreshold, expiresAt.
        Raises KeyError if the key doesn't exist.
        """
        async with self._lock:
            record = self._records.get(key)
            if not record:
                raise KeyError(f"Key not found: {key[:8]}...")

            if "name" in updates:
                record.name = str(updates["name"])
            if "enabled" in updates:
                record.enabled = bool(updates["enabled"])
            if "tokenQuota" in updates:
                record.tokenQuota = max(0, int(updates["tokenQuota"]))
            if "creditsThreshold" in updates:
                record.creditsThreshold = max(0, int(updates["creditsThreshold"]))
            if "expiresAt" in updates:
                record.expiresAt = max(0, int(updates["expiresAt"]))

            await self._save()
            logger.info(f"Updated API key '{record.name}'")
            return record

    async def delete_key(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        async with self._lock:
            if key not in self._records:
                return False
            del self._records[key]
            await self._save()
            logger.info(f"Deleted API key {key[:8]}...")
            return True

    async def reset_usage(self, key: str) -> APIKeyRecord:
        """Reset usage counters (tokensUsed, requestsUsed) for a key."""
        async with self._lock:
            record = self._records.get(key)
            if not record:
                raise KeyError(f"Key not found: {key[:8]}...")
            record.tokensUsed = 0
            record.requestsUsed = 0
            await self._save()
            logger.info(f"Reset usage for API key '{record.name}'")
            return record
