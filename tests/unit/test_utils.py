# -*- coding: utf-8 -*-

"""
Unit tests for utils.py - Conversation cache affinity helpers.

Kiro applies an implicit server-side prefix cache keyed by
conversationState.conversationId. These tests pin down how the gateway derives
that id:

- a client-provided session identity is preferred when it is usable
- otherwise a hash of the conversation's FIRST message is used, so the id stays
  stable as the conversation grows
- the CONVERSATION_CACHE_AFFINITY toggle restores the legacy random-UUID behavior
"""

import re
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import kiro.utils as utils


def _request(metadata=None, user=None, messages=None, input_value=None):
    """Builds a minimal stand-in for any of the three request models."""
    attrs = {"metadata": metadata, "user": user}
    if messages is not None:
        attrs["messages"] = messages
    if input_value is not None:
        attrs["input"] = input_value
    return SimpleNamespace(**attrs)


def _user_msg(content="Hello"):
    return {"role": "user", "content": content}


# =============================================================================
# Client session identity extraction
# =============================================================================

class TestResolveConversationIdClientSessionId:
    """Tests for preferring a client-provided session identity."""

    def test_anthropic_json_metadata_session_id(self):
        """
        What it does: Extracts session_id from a JSON-encoded metadata.user_id.
        Purpose: Claude Code sends its stable session id this way.
        """
        print("Action: resolve with a Claude Code style metadata.user_id...")
        request = _request(
            metadata={
                "user_id": '{"device_id":"d","account_uuid":"","session_id":"44a858b6-e221-4313-aeca-116d4ee8f157"}'
            },
            messages=[_user_msg()],
        )
        result = utils.resolve_conversation_id(request)
        print(f"Comparing: Expected '44a858b6-...', Got {result!r}")
        assert result == "44a858b6-e221-4313-aeca-116d4ee8f157"

    def test_explicit_metadata_session_id(self):
        """
        What it does: Uses a direct metadata["session_id"] value.
        Purpose: Some clients send the field unwrapped.
        """
        print("Action: resolve with metadata={'session_id': 's1'}...")
        result = utils.resolve_conversation_id(_request(metadata={"session_id": "s1"}, messages=[_user_msg()]))
        print(f"Comparing: Expected 's1', Got {result!r}")
        assert result == "s1"

    def test_openai_user_field(self):
        """
        What it does: Uses the OpenAI-style `user` field as the identity.
        Purpose: Cover clients that have no metadata field at all.
        """
        print("Action: resolve with user='sess-openai'...")
        result = utils.resolve_conversation_id(_request(user="sess-openai", messages=[_user_msg()]))
        print(f"Comparing: Expected 'sess-openai', Got {result!r}")
        assert result == "sess-openai"

    def test_metadata_takes_priority_over_user(self):
        """
        What it does: Prefers metadata over the user field when both are present.
        Purpose: Keep one deterministic priority order.
        """
        print("Action: resolve with both metadata and user...")
        result = utils.resolve_conversation_id(
            _request(metadata={"session_id": "from-metadata"}, user="from-user", messages=[_user_msg()])
        )
        print(f"Comparing: Expected 'from-metadata', Got {result!r}")
        assert result == "from-metadata"

    def test_plain_non_json_user_id_is_usable(self):
        """
        What it does: Accepts a plain (non-JSON) metadata.user_id string.
        Purpose: Simpler clients send a bare id, not a JSON blob.
        """
        print("Action: resolve with user_id='abc-123'...")
        result = utils.resolve_conversation_id(_request(metadata={"user_id": "abc-123"}, messages=[_user_msg()]))
        print(f"Comparing: Expected 'abc-123', Got {result!r}")
        assert result == "abc-123"


class TestResolveConversationIdClientSessionIdEdgeCases:
    """Edge cases for identity extraction - every one must fall back to the hash."""

    def _expected_hash(self, messages):
        return utils._hash_first_message(messages)

    def test_malformed_json_falls_back_to_hash(self):
        """
        What it does: Ignores a metadata.user_id that is broken JSON.
        Purpose: A broken identity must never become a per-request-varying key.
        """
        print("Action: resolve with malformed JSON in metadata.user_id...")
        messages = [_user_msg()]
        request = _request(metadata={"user_id": '{"session_id":'}, messages=messages)
        result = utils.resolve_conversation_id(request)
        print(f"Comparing: Expected hash {self._expected_hash(messages)!r}, Got {result!r}")
        assert result == self._expected_hash(messages)

    def test_json_without_session_id_falls_back_to_hash(self):
        """
        What it does: Does not use device-scoped fields from the JSON blob.
        Purpose: device_id/account_uuid would collapse unrelated conversations.
        """
        print("Action: resolve with only device_id/account_uuid present...")
        messages = [_user_msg()]
        request = _request(
            metadata={"user_id": '{"device_id":"dev-1","account_uuid":"acct-1"}'},
            messages=messages,
        )
        result = utils.resolve_conversation_id(request)
        print(f"Comparing: Expected hash {self._expected_hash(messages)!r}, Got {result!r}")
        assert result == self._expected_hash(messages)
        assert result != "dev-1"

    @pytest.mark.parametrize("value", [12345, {"a": 1}, ["x"], True])
    def test_non_string_user_id_ignored(self, value):
        """
        What it does: Ignores non-string metadata.user_id values.
        Purpose: Typed metadata must not crash the derivation.
        """
        print(f"Action: resolve with non-string user_id={value!r}...")
        messages = [_user_msg()]
        result = utils.resolve_conversation_id(_request(metadata={"user_id": value}, messages=messages))
        print(f"Comparing: Expected hash, Got {result!r}")
        assert result == self._expected_hash(messages)

    @pytest.mark.parametrize("value", ["", "   ", "\t"])
    def test_empty_and_whitespace_fall_back(self, value):
        """
        What it does: Rejects empty/whitespace-only identities.
        Purpose: An empty key is not an identity.
        """
        print(f"Action: resolve with user={value!r}...")
        messages = [_user_msg()]
        result = utils.resolve_conversation_id(_request(metadata={"session_id": value}, user=value, messages=messages))
        print(f"Comparing: Expected hash, Got {result!r}")
        assert result == self._expected_hash(messages)

    @pytest.mark.parametrize("value", ["user", "test", "test_user", "testuser", "anonymous", "none", "null", "unknown", "default", "guest", "TEST_USER"])
    def test_placeholder_values_rejected(self, value):
        """
        What it does: Rejects generic placeholder identities.
        Purpose: Placeholders would key every client onto one cache entry.
        """
        print(f"Action: resolve with placeholder user={value!r}...")
        messages = [_user_msg()]
        result = utils.resolve_conversation_id(_request(metadata={"user_id": value}, messages=messages))
        print(f"Comparing: Expected hash, Got {result!r}")
        assert result == self._expected_hash(messages)

    def test_overlong_id_falls_back_without_truncation(self):
        """
        What it does: Rejects (rather than truncates) an over-length identity.
        Purpose: A truncated key is a surprising identity; the hash is better.
        """
        print("Action: resolve with a 200-character user id...")
        long_id = "a" * 200
        messages = [_user_msg()]
        result = utils.resolve_conversation_id(_request(user=long_id, messages=messages))
        print(f"Comparing: Expected hash, Got {result!r}")
        assert result == self._expected_hash(messages)
        assert result != long_id[:128]

    @pytest.mark.parametrize("value", ["sess\n1", "sess\x001", "sess\r1"])
    def test_control_characters_rejected(self, value):
        """
        What it does: Rejects identities containing control characters.
        Purpose: Control chars can confuse payload/header handling downstream.
        """
        print(f"Action: resolve with control characters in {value!r}...")
        messages = [_user_msg()]
        result = utils.resolve_conversation_id(_request(user=value, messages=messages))
        print(f"Comparing: Expected hash, Got {result!r}")
        assert result == self._expected_hash(messages)

    def test_boundary_length_is_accepted(self):
        """
        What it does: Accepts an identity exactly at the length limit.
        Purpose: Pin the boundary so it cannot silently regress.
        """
        print("Action: resolve with a 128-character user id...")
        boundary_id = "b" * utils._MAX_CONVERSATION_ID_LENGTH
        result = utils.resolve_conversation_id(_request(user=boundary_id, messages=[_user_msg()]))
        print(f"Comparing: Expected the id itself, Got {result!r}")
        assert result == boundary_id


# =============================================================================
# First-message hash fallback
# =============================================================================

class TestResolveConversationIdHashFallback:
    """Tests for the first-message hash used when no identity is available."""

    def test_stable_across_turns(self):
        """
        What it does: Same conversation, one more exchange -> same id.
        Purpose: Core regression - the last message must never affect the key.
        """
        print("Action: resolving turn 1 and turn 2 of the same conversation...")
        turn_1 = [_user_msg("A"), {"role": "assistant", "content": "B"}]
        turn_2 = turn_1 + [_user_msg("C"), {"role": "assistant", "content": "D"}]
        first = utils.resolve_conversation_id(_request(messages=turn_1))
        second = utils.resolve_conversation_id(_request(messages=turn_2))
        print(f"Comparing: turn1={first!r}, turn2={second!r}")
        assert first == second

    def test_independent_of_last_message(self):
        """
        What it does: Appending messages leaves the id untouched.
        Purpose: Proves the old messages[-1] instability is gone.
        """
        print("Action: comparing [A,B] with [A,B,C]...")
        short = [_user_msg("A"), {"role": "assistant", "content": "B"}]
        long = short + [_user_msg("C")]
        print(f"Comparing: {utils._hash_first_message(short)!r} vs {utils._hash_first_message(long)!r}")
        assert utils._hash_first_message(short) == utils._hash_first_message(long)

    def test_differs_for_different_first_messages(self):
        """
        What it does: Different conversations produce different ids.
        Purpose: Two conversations must not share one cache key.
        """
        print("Action: resolving two conversations with different openings...")
        first = utils.resolve_conversation_id(_request(messages=[_user_msg("Alpha")]))
        second = utils.resolve_conversation_id(_request(messages=[_user_msg("Beta")]))
        print(f"Comparing: {first!r} vs {second!r}")
        assert first != second

    def test_hash_is_16_hex_characters(self):
        """
        What it does: Verifies the fallback id shape.
        Purpose: Keep the id format stable and short.
        """
        print("Action: resolving a hash-derived id...")
        result = utils.resolve_conversation_id(_request(messages=[_user_msg("shape check")]))
        print(f"Comparing: {result!r} against ^[0-9a-f]{{16}}$")
        assert re.fullmatch(r"[0-9a-f]{16}", result)

    def test_handles_anthropic_content_block_lists(self):
        """
        What it does: Hashes Anthropic-style content block lists deterministically.
        Purpose: Real Anthropic messages carry block lists, not plain strings.
        """
        print("Action: resolving with content blocks...")
        blocks = [{"type": "text", "text": "hi"}, {"type": "tool_result", "content": "x"}]
        first = utils.resolve_conversation_id(_request(messages=[{"role": "user", "content": blocks}]))
        second = utils.resolve_conversation_id(_request(messages=[{"role": "user", "content": blocks}]))
        print(f"Comparing: {first!r} vs {second!r}")
        assert first == second
        assert re.fullmatch(r"[0-9a-f]{16}", first)

    def test_handles_pydantic_like_message_objects(self):
        """
        What it does: Hashes message objects exposing role/content attributes.
        Purpose: Anthropic/OpenAI routers pass pydantic models, not dicts.
        """
        print("Action: resolving with a pydantic-like message object...")

        class _Msg:
            role = "user"
            content = "object based"

        first = utils.resolve_conversation_id(_request(messages=[_Msg()]))
        second = utils.resolve_conversation_id(_request(messages=[_Msg()]))
        print(f"Comparing: {first!r} vs {second!r}")
        assert first == second
        assert re.fullmatch(r"[0-9a-f]{16}", first)

    def test_handles_responses_bare_string_input(self):
        """
        What it does: Hashes the Responses bare-string `input` form.
        Purpose: /v1/responses accepts a plain string as its input.
        """
        print("Action: resolving with input as a bare string...")
        first = utils.resolve_conversation_id(_request(input_value="hello there"))
        second = utils.resolve_conversation_id(_request(input_value="hello there"))
        print(f"Comparing: {first!r} vs {second!r}")
        assert first == second
        assert re.fullmatch(r"[0-9a-f]{16}", first)

    def test_handles_responses_item_list(self):
        """
        What it does: Hashes the Responses typed-item list form.
        Purpose: /v1/responses also accepts a list of typed items.
        """
        print("Action: resolving with input as a list of items...")
        items = [{"role": "user", "content": [{"type": "input_text", "text": "hi"}]}]
        result = utils.resolve_conversation_id(_request(input_value=items))
        print(f"Comparing: {result!r}")
        assert result == utils._hash_first_message(items)
        assert re.fullmatch(r"[0-9a-f]{16}", result)

    @pytest.mark.parametrize("messages", [None, [], ""])
    def test_no_usable_messages_returns_random_uuid(self, messages):
        """
        What it does: Falls back to a random UUID when nothing can be hashed.
        Purpose: Never return an empty or shared key.
        """
        print(f"Action: resolving with messages={messages!r}...")
        first = utils.resolve_conversation_id(_request(messages=messages))
        second = utils.resolve_conversation_id(_request(messages=messages))
        print(f"Comparing: {first!r} vs {second!r}")
        assert first != second
        assert len(first) == 36

    def test_falls_back_to_request_attributes_when_messages_omitted(self):
        """
        What it does: Reads request.messages / request.input when messages is not passed.
        Purpose: Routers may rely on the implicit lookup.
        """
        print("Action: resolving without passing messages explicitly...")
        via_messages = utils.resolve_conversation_id(_request(messages=[_user_msg("implicit")]))
        via_input = utils.resolve_conversation_id(_request(input_value=[_user_msg("implicit")]))
        print(f"Comparing: {via_messages!r} vs {via_input!r}")
        assert via_messages == via_input


# =============================================================================
# Toggle
# =============================================================================

class TestResolveConversationIdToggle:
    """Tests for the CONVERSATION_CACHE_AFFINITY switch."""

    def test_enabled_by_default(self):
        """
        What it does: Confirms the toggle defaults to enabled.
        Purpose: The cost saving must be on out of the box.
        """
        print(f"Checking default: CONVERSATION_CACHE_AFFINITY={utils.CONVERSATION_CACHE_AFFINITY}")
        assert utils.CONVERSATION_CACHE_AFFINITY is True

    def test_disabled_returns_random_uuid_per_call(self):
        """
        What it does: Reproduces the legacy random-UUID behavior when disabled.
        Purpose: The switch must fully restore pre-change behavior.
        """
        print("Action: resolving twice with the toggle disabled...")
        request = _request(metadata={"session_id": "sess-1"}, messages=[_user_msg()])
        with patch.object(utils, "CONVERSATION_CACHE_AFFINITY", False):
            first = utils.resolve_conversation_id(request)
            second = utils.resolve_conversation_id(request)
        print(f"Comparing: {first!r} vs {second!r}")
        assert first != second
        assert len(first) == 36

    def test_disabled_ignores_available_session_id(self):
        """
        What it does: Ensures the toggle wins over an available session id.
        Purpose: No upstream cache affinity must leak when disabled.
        """
        print("Action: resolving with a session id while the toggle is disabled...")
        request = _request(metadata={"session_id": "sess-1"}, messages=[_user_msg()])
        with patch.object(utils, "CONVERSATION_CACHE_AFFINITY", False):
            result = utils.resolve_conversation_id(request)
        print(f"Comparing: Expected a random UUID, Got {result!r}")
        assert result != "sess-1"
        assert len(result) == 36
