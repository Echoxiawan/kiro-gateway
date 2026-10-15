# -*- coding: utf-8 -*-

# Kiro Gateway
# https://github.com/Echoxiawan/kiro-gateway
# Copyright (C) 2025 Jwadow
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
Utility functions for Kiro Gateway.

Contains functions for fingerprint generation, header formatting,
and other common utilities.
"""

import hashlib
import json
import uuid
from typing import TYPE_CHECKING, List, Dict, Any, Optional

from loguru import logger

from kiro.config import CONVERSATION_CACHE_AFFINITY, KIRO_IDE_VERSION

if TYPE_CHECKING:
    from kiro.auth import KiroAuthManager


def get_machine_fingerprint() -> str:
    """
    Generates a unique machine fingerprint based on hostname and username.
    
    Used for User-Agent formation to identify a specific gateway installation.
    
    Returns:
        SHA256 hash of the string "{hostname}-{username}-kiro-gateway"
    """
    try:
        import socket
        import getpass
        
        hostname = socket.gethostname()
        username = getpass.getuser()
        unique_string = f"{hostname}-{username}-kiro-gateway"
        
        return hashlib.sha256(unique_string.encode()).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to get machine fingerprint: {e}")
        return hashlib.sha256(b"default-kiro-gateway").hexdigest()


def get_kiro_headers(auth_manager: "KiroAuthManager", token: str) -> dict:
    """
    Builds headers for Kiro API requests.
    
    Includes all necessary headers for authentication and identification:
    - Authorization with Bearer token
    - User-Agent with fingerprint
    - AWS CodeWhisperer specific headers
    
    Args:
        auth_manager: Authentication manager for obtaining fingerprint
        token: Access token for authorization
    
    Returns:
        Dictionary with headers for HTTP request
    """
    fingerprint = auth_manager.fingerprint
    
    return {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-amz-json-1.0",
        "x-amz-target": "AmazonCodeWhispererStreamingService.GenerateAssistantResponse",
        "User-Agent": f"aws-sdk-js/1.0.27 ua/2.1 os/win32#10.0.19044 lang/js md/nodejs#22.21.1 api/codewhispererstreaming#1.0.27 m/E KiroIDE-{KIRO_IDE_VERSION}-{fingerprint}",
        "x-amz-user-agent": f"aws-sdk-js/1.0.27 KiroIDE-{KIRO_IDE_VERSION}-{fingerprint}",
        "x-amzn-codewhisperer-optout": "true",
        "x-amzn-kiro-agent-mode": "vibe",
        "amz-sdk-invocation-id": str(uuid.uuid4()),
        "amz-sdk-request": "attempt=1; max=3",
    }


def generate_completion_id() -> str:
    """
    Generates a unique ID for chat completion.
    
    Returns:
        ID in format "chatcmpl-{uuid_hex}"
    """
    return f"chatcmpl-{uuid.uuid4().hex}"


# ---------------------------------------------------------------------------
# Conversation cache affinity
# ---------------------------------------------------------------------------
#
# Kiro applies an implicit server-side prefix cache keyed by
# conversationState.conversationId. Reusing one stable id across the turns of a
# single client conversation turns the already-sent prefix into a cache read
# (measured roughly 10x fewer credits) instead of billing the whole prefix again
# on every turn.
#
# The AWS-style `cachePoint` field exists in the upstream SDK schema but is a
# no-op on this endpoint, so it is deliberately never sent.

# Longest client-supplied session id accepted as-is. Longer values are rejected
# (and the first-message hash is used instead) rather than truncated, so the id
# sent upstream always equals an identity the client actually provided.
_MAX_CONVERSATION_ID_LENGTH = 128

# Values that are placeholders rather than real session identities. Using one as
# an affinity key would collapse unrelated conversations onto a single cache key.
_PLACEHOLDER_SESSION_IDS = frozenset({
    "user", "test", "test_user", "testuser", "anonymous",
    "none", "null", "unknown", "default", "guest",
})


def _is_usable_session_id(value: Any) -> Optional[str]:
    """
    Validates a candidate client session id.

    Args:
        value: Candidate value from request metadata (any type)

    Returns:
        The stripped id when it is usable, otherwise None
    """
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    if not candidate or len(candidate) > _MAX_CONVERSATION_ID_LENGTH:
        return None
    if candidate.lower() in _PLACEHOLDER_SESSION_IDS:
        return None
    # Control characters are never legitimate in an id and could confuse
    # payload/header handling downstream.
    if any(ord(ch) < 0x20 for ch in candidate):
        return None
    return candidate


def _extract_client_session_id(metadata: Any, user: Any) -> Optional[str]:
    """
    Extracts a client-provided conversation identity from request metadata.

    Sources, in priority order:
      1. metadata["session_id"]                (explicit field)
      2. metadata["user_id"] parsed as JSON    -> its "session_id"
      3. metadata["user_id"] as a plain string (non-JSON clients)
      4. user                                  (OpenAI / Responses `user` field)

    Only `session_id` is read from the parsed JSON. `device_id` and
    `account_uuid` are deliberately ignored: they are device-scoped rather than
    conversation-scoped, so keying on them would collapse unrelated
    conversations onto one cache entry.

    A malformed value never raises - it simply falls through to the next
    candidate, because a broken identity is not trustworthy as a cache key.

    Args:
        metadata: Request metadata dict (Anthropic / Responses), or anything else
        user: `user` field from OpenAI / Responses requests

    Returns:
        A validated non-empty id, or None when no usable identity is present
    """
    candidates: List[Any] = []

    if isinstance(metadata, dict):
        candidates.append(metadata.get("session_id"))

        raw_user_id = metadata.get("user_id")
        if isinstance(raw_user_id, str):
            parsed = None
            try:
                parsed = json.loads(raw_user_id)
            except (ValueError, TypeError):
                parsed = None

            if isinstance(parsed, dict):
                # Claude Code wraps its identity in a JSON-encoded string.
                candidates.append(parsed.get("session_id"))
            elif not raw_user_id.lstrip().startswith(("{", "[")):
                # Plain (non-JSON) user_id from simpler clients.
                candidates.append(raw_user_id)

    candidates.append(user)

    for candidate in candidates:
        usable = _is_usable_session_id(candidate)
        if usable:
            return usable
    return None


def _normalize_message_content(content: Any) -> str:
    """
    Serializes message content deterministically for hashing.

    Handles plain strings, Anthropic-style content block lists (which may hold
    pydantic models) and Responses-style typed items.

    Args:
        content: Message content in any supported shape

    Returns:
        A deterministic string representation
    """
    if isinstance(content, str):
        return content

    def _fallback(obj: Any) -> Any:
        model_dump = getattr(obj, "model_dump", None)
        if callable(model_dump):
            return model_dump()
        return str(obj)

    if hasattr(content, "model_dump") or isinstance(content, (dict, list)):
        try:
            return json.dumps(content, sort_keys=True, default=_fallback)
        except (TypeError, ValueError):
            return str(content)
    return str(content)


def _hash_first_message(messages: Any) -> Optional[str]:
    """
    Derives a stable cache-affinity id from the first message of a conversation.

    Only the first message is used, so the id stays identical as the
    conversation grows. The last message must never be part of the key: it
    changes on every turn and would defeat cache affinity.

    Note: if a client mutates its first message between turns (for example when
    it embeds a volatile system reminder), affinity is lost - the only
    consequence is a cache miss, never an error.

    Args:
        messages: Message list (pydantic models, dicts, or Responses items),
            or a bare string for the Responses `input` string form

    Returns:
        16-char hex digest, or None when there is no usable first message
    """
    if messages is None:
        return None

    if isinstance(messages, str):
        if not messages:
            return None
        first: Any = {"role": "user", "content": messages}
    else:
        try:
            items = list(messages)
        except TypeError:
            return None
        if not items:
            return None
        first = items[0]

    if isinstance(first, dict):
        role = first.get("role", "user")
        content = first.get("content")
        if content is None:
            content = first.get("text", "")
    else:
        role = getattr(first, "role", "user")
        content = getattr(first, "content", None)
        if content is None:
            content = getattr(first, "text", "")

    payload = json.dumps(
        {"role": role, "content": _normalize_message_content(content)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def resolve_conversation_id(request: Any, messages: Any = None) -> str:
    """
    Returns the conversationId to send to Kiro for upstream prefix-cache affinity.

    Priority:
      1. CONVERSATION_CACHE_AFFINITY disabled -> fresh random UUID (legacy behavior).
      2. Client-provided session id (see _extract_client_session_id).
      3. SHA-256 prefix of the conversation's first message.
      4. Random UUID (no identity and no messages available).

    Args:
        request: Parsed request object (any of the three API shapes). Read
            defensively via getattr so this module stays decoupled from the models.
        messages: Conversation messages for the hash fallback. Routers pass
            request_data.messages (Anthropic / OpenAI) or request_data.input
            (Responses, which has no `messages` field).

    Returns:
        A stable id string, or a random UUID when affinity is disabled or the
        identity cannot be determined

    Example:
        >>> request = SimpleNamespace(metadata={"session_id": "sess-1"})
        >>> resolve_conversation_id(request)
        'sess-1'
    """
    if not CONVERSATION_CACHE_AFFINITY:
        # Legacy behavior: a fresh id every request, so no upstream cache reuse.
        return str(uuid.uuid4())

    session_id = _extract_client_session_id(
        getattr(request, "metadata", None),
        getattr(request, "user", None),
    )
    if session_id:
        return session_id

    if messages is None:
        messages = getattr(request, "messages", None)
        if messages is None:
            messages = getattr(request, "input", None)

    return _hash_first_message(messages) or str(uuid.uuid4())


def generate_tool_call_id() -> str:
    """
    Generates a unique ID for tool call.
    
    Returns:
        ID in format "call_{uuid_hex[:8]}"
    """
    return f"call_{uuid.uuid4().hex[:8]}"