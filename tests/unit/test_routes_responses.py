# -*- coding: utf-8 -*-

"""
Unit tests for the OpenAI Responses API endpoint (routes_responses.py).

Tests the following endpoint:
- POST /v1/responses - Responses API

Note: `ResponsesRequest` has no `messages` field - its conversation input lives
in `input` (`Union[str, List[Any]]`). The conversation cache affinity tests here
pin that down, because relying on `messages` would silently disable upstream
cache reuse for this endpoint.

For OpenAI Chat Completions tests, see test_routes_openai.py.
For Anthropic tests, see test_routes_anthropic.py.
"""

from unittest.mock import patch

import pytest

from kiro.routes_responses import router  # noqa: F401  (imports the module under test)


def _capture_responses_conversation_id(client, api_key, payload):
    """
    Sends a /v1/responses request and returns the conversationId the route
    passed to the Kiro payload builder. The builder is patched, so no upstream
    request is made.
    """
    captured = {}

    def _fake_build(request_data, conversation_id, profile_arn):
        captured["conversation_id"] = conversation_id
        return {"conversationState": {"conversationId": conversation_id}}

    with patch("kiro.routes_responses.build_kiro_payload", side_effect=_fake_build):
        client.post(
            "/v1/responses",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload,
        )
    return captured.get("conversation_id")


class TestResponsesConversationIdAffinity:
    """Tests that /v1/responses reuses a stable upstream cache-affinity key."""

    def test_payload_carries_metadata_session_id(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies metadata.user_id session_id becomes the Kiro conversationId.
        Purpose: This is what makes Kiro's server-side prefix cache hit.
        """
        print("Action: POST /v1/responses with metadata.user_id session_id...")
        conversation_id = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {
                "model": "claude-sonnet-4-5",
                "input": "Hello there",
                "metadata": {"user_id": '{"session_id":"sess-responses"}'},
            },
        )
        print(f"Comparing: Expected 'sess-responses', Got {conversation_id!r}")
        assert conversation_id == "sess-responses"

    def test_payload_carries_user_field(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies the Responses `user` field becomes the conversationId.
        Purpose: Cover the second identity source this endpoint exposes.
        """
        print("Action: POST /v1/responses with user='sess-user-field'...")
        conversation_id = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {
                "model": "claude-sonnet-4-5",
                "input": "Hello there",
                "user": "sess-user-field",
            },
        )
        print(f"Comparing: Expected 'sess-user-field', Got {conversation_id!r}")
        assert conversation_id == "sess-user-field"

    def test_hash_from_bare_string_input(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies the hash fallback reads `input` when it is a bare string.
        Purpose: Responses has no `messages`; the route must pass `input`.
        """
        print("Action: POST /v1/responses with a bare string input...")
        first = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {"model": "claude-sonnet-4-5", "input": "Hello there"},
        )
        second = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {"model": "claude-sonnet-4-5", "input": "Hello there"},
        )
        print(f"Comparing: {first!r} vs {second!r}")
        assert first is not None
        assert len(first) == 16
        assert first == second

    def test_hash_from_item_list_input(self, test_client, valid_proxy_api_key):
        """
        What it does: Verifies the hash fallback reads `input` when it is an item list.
        Purpose: The other documented shape of the Responses input field.
        """
        print("Action: POST /v1/responses with a typed item list input...")
        items = [{"role": "user", "content": [{"type": "input_text", "text": "Hello there"}]}]
        first = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {"model": "claude-sonnet-4-5", "input": items},
        )
        second = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {"model": "claude-sonnet-4-5", "input": items},
        )
        print(f"Comparing: {first!r} vs {second!r}")
        assert first is not None
        assert len(first) == 16
        assert first == second

    def test_user_without_messages_field_still_hits_hash_path(self, test_client, valid_proxy_api_key):
        """
        What it does: Guards against silently reading a non-existent `messages` attribute.
        Purpose: If the route passed request_data.messages, this would be a random UUID.
        """
        print("Action: POST /v1/responses with no identity at all...")
        first = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {"model": "claude-sonnet-4-5", "input": "Hello there"},
        )
        print(f"Comparing: Expected a 16-char hash, Got {first!r}")
        # A random UUID would be 36 characters; the hash path is 16.
        assert len(first) == 16

    def test_payload_carries_codex_prompt_cache_key(self, test_client, valid_proxy_api_key):
        """
        What it does: Uses Codex prompt_cache_key as the Kiro conversationId.
        Purpose: This is Codex's explicit stable cache-affinity identity.
        """
        print("Action: POST /v1/responses with Codex prompt_cache_key...")
        conversation_id = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {
                "model": "claude-sonnet-4-5",
                "input": [{"type": "additional_tools", "role": "developer", "tools": []}, {"type": "message", "role": "user", "content": "Hello"}],
                "prompt_cache_key": "codex-prompt-cache-key",
                "client_metadata": {"session_id": "codex-session", "thread_id": "codex-thread"},
            },
        )
        print(f"Comparing: Expected 'codex-prompt-cache-key', Got {conversation_id!r}")
        assert conversation_id == "codex-prompt-cache-key"

    def test_payload_carries_codex_client_metadata_session_id(self, test_client, valid_proxy_api_key):
        """
        What it does: Uses client_metadata.session_id when Codex omits prompt_cache_key.
        Purpose: Preserve cache affinity across supported Codex request variants.
        """
        print("Action: POST /v1/responses with Codex client_metadata.session_id...")
        conversation_id = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "client_metadata": {"session_id": "codex-session", "thread_id": "codex-thread"},
            },
        )
        print(f"Comparing: Expected 'codex-session', Got {conversation_id!r}")
        assert conversation_id == "codex-session"

    def test_payload_carries_codex_client_metadata_thread_id(self, test_client, valid_proxy_api_key):
        """
        What it does: Uses client_metadata.thread_id when session_id is unavailable.
        Purpose: Keep a stable final Codex-specific fallback.
        """
        print("Action: POST /v1/responses with Codex client_metadata.thread_id...")
        conversation_id = _capture_responses_conversation_id(
            test_client,
            valid_proxy_api_key,
            {
                "model": "claude-sonnet-4-5",
                "input": "Hello",
                "client_metadata": {"thread_id": "codex-thread"},
            },
        )
        print(f"Comparing: Expected 'codex-thread', Got {conversation_id!r}")
        assert conversation_id == "codex-thread"

    def test_codex_affinity_log_does_not_expose_raw_key(self, test_client, valid_proxy_api_key):
        """
        What it does: Logs only the affinity source label for a Codex request.
        Purpose: Operators can diagnose cache routing without leaking session keys.
        """
        print("Action: POST /v1/responses while capturing route debug messages...")
        raw_key = "codex-secret-session-key"
        captured = {}

        def _fake_build(request_data, conversation_id, profile_arn):
            captured["conversation_id"] = conversation_id
            return {"conversationState": {"conversationId": conversation_id}}

        with (
            patch("kiro.routes_responses.build_kiro_payload", side_effect=_fake_build),
            patch("kiro.routes_responses.logger.debug") as debug_log,
        ):
            test_client.post(
                "/v1/responses",
                headers={"Authorization": f"Bearer {valid_proxy_api_key}"},
                json={
                    "model": "claude-sonnet-4-5",
                    "input": "Hello",
                    "prompt_cache_key": raw_key,
                },
            )

        messages = [str(call.args[0]) for call in debug_log.call_args_list if call.args]
        print(f"Captured debug messages: {messages!r}")
        assert captured["conversation_id"] == raw_key
        assert "Conversation cache affinity source: codex_prompt_cache_key" in messages
        assert all(raw_key not in message for message in messages)
