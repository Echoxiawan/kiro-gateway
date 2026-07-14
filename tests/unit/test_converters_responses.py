# -*- coding: utf-8 -*-

"""
Unit tests for converters_responses module (OpenAI Responses API / Codex CLI).

Tests for Responses-specific conversion logic:
- Converting Responses `input` items to unified format
- Converting Responses tools (flat format) to unified format
- Reasoning effort -> thinking config
- Building Kiro payload end to end
"""

import json

import pytest

from kiro.converters_responses import (
    build_kiro_payload,
    convert_responses_input_to_unified,
    convert_responses_tools_to_unified,
    extract_thinking_config_from_responses,
    _normalize_content_parts,
)
from kiro.models_responses import ResponsesRequest, ResponsesTool


# ==================================================================================================
# convert_responses_input_to_unified
# ==================================================================================================

class TestConvertResponsesInput:
    """Tests for convert_responses_input_to_unified."""

    def test_plain_string_input_becomes_user_message(self):
        """
        What it does: A bare string `input` becomes a single user message.
        Purpose: Codex may send simple string prompts.
        """
        req = ResponsesRequest(model="claude-sonnet-4.5", input="Hello there")
        system, unified = convert_responses_input_to_unified(req)

        assert system == ""
        assert len(unified) == 1
        assert unified[0].role == "user"
        assert unified[0].content == "Hello there"

    def test_instructions_become_system_prompt(self):
        """
        What it does: `instructions` is extracted as the system prompt.
        Purpose: Responses uses instructions instead of a system message.
        """
        req = ResponsesRequest(
            model="m", instructions="You are Codex", input="hi"
        )
        system, unified = convert_responses_input_to_unified(req)

        assert system == "You are Codex"
        assert len(unified) == 1

    def test_message_items_with_typed_content_parts(self):
        """
        What it does: message items with input_text/output_text parts are normalized.
        Purpose: Responses content parts differ from Chat Completions.
        """
        req = ResponsesRequest(model="m", input=[
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "question"}]},
            {"type": "message", "role": "assistant",
             "content": [{"type": "output_text", "text": "answer"}]},
        ])
        system, unified = convert_responses_input_to_unified(req)

        assert len(unified) == 2
        assert unified[0].role == "user"
        assert unified[1].role == "assistant"

    def test_system_and_developer_messages_fold_into_system_prompt(self):
        """
        What it does: system/developer input messages join the system prompt.
        Purpose: Kiro only supports user/assistant history.
        """
        req = ResponsesRequest(model="m", instructions="Base", input=[
            {"type": "message", "role": "developer",
             "content": [{"type": "input_text", "text": "Dev rule"}]},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
        ])
        system, unified = convert_responses_input_to_unified(req)

        assert "Base" in system
        assert "Dev rule" in system
        assert len(unified) == 1
        assert unified[0].role == "user"

    def test_function_call_becomes_assistant_tool_call(self):
        """
        What it does: a function_call item becomes an assistant message with tool_calls.
        Purpose: Responses tool calls are top-level items.
        """
        req = ResponsesRequest(model="m", input=[
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "run ls"}]},
            {"type": "function_call", "call_id": "call_1", "name": "shell",
             "arguments": '{"cmd":"ls"}'},
        ])
        system, unified = convert_responses_input_to_unified(req)

        assistant_msgs = [m for m in unified if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        tc = assistant_msgs[0].tool_calls[0]
        assert tc["id"] == "call_1"
        assert tc["function"]["name"] == "shell"
        assert tc["function"]["arguments"] == '{"cmd":"ls"}'

    def test_function_call_output_becomes_user_tool_result(self):
        """
        What it does: a function_call_output item becomes a user message tool_result.
        Purpose: call_id must be preserved to pair with the tool call.
        """
        req = ResponsesRequest(model="m", input=[
            {"type": "function_call_output", "call_id": "call_1", "output": "file1.txt"},
        ])
        system, unified = convert_responses_input_to_unified(req)

        tr = unified[0].tool_results[0]
        assert tr["tool_use_id"] == "call_1"
        assert tr["content"] == "file1.txt"

    def test_reasoning_items_are_ignored(self):
        """
        What it does: reasoning items (rs_...) are dropped, not passed through.
        Purpose: Kiro has no encrypted reasoning; passing rs_ ids back errors.
        """
        req = ResponsesRequest(model="m", input=[
            {"type": "reasoning", "id": "rs_abc", "encrypted_content": "xxxx",
             "summary": []},
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "hi"}]},
        ])
        system, unified = convert_responses_input_to_unified(req)

        assert len(unified) == 1
        assert unified[0].role == "user"

    def test_function_call_dict_arguments_serialized(self):
        """
        What it does: object arguments are serialized to a JSON string.
        Purpose: Some clients send arguments as objects, Kiro expects strings.
        """
        req = ResponsesRequest(model="m", input=[
            {"type": "function_call", "call_id": "c", "name": "f",
             "arguments": {"a": 1}},
        ])
        _, unified = convert_responses_input_to_unified(req)
        args = unified[0].tool_calls[0]["function"]["arguments"]
        assert json.loads(args) == {"a": 1}


# ==================================================================================================
# _normalize_content_parts
# ==================================================================================================

class TestNormalizeContentParts:
    def test_input_image_becomes_image_url_block(self):
        """
        What it does: input_image with a data URL becomes an image_url block.
        Purpose: core extract_images_from_content understands image_url.
        """
        parts = _normalize_content_parts([
            {"type": "input_image", "image_url": "data:image/png;base64,ABC"},
        ])
        assert parts[0]["type"] == "image_url"
        assert parts[0]["image_url"]["url"] == "data:image/png;base64,ABC"

    def test_string_passthrough(self):
        assert _normalize_content_parts("hi") == "hi"

    def test_none_becomes_empty(self):
        assert _normalize_content_parts(None) == ""


# ==================================================================================================
# convert_responses_tools_to_unified
# ==================================================================================================

class TestConvertResponsesTools:
    def test_flat_function_tool(self):
        """
        What it does: flat-format function tool converts to a UnifiedTool.
        Purpose: Responses tools have no nested `function` object.
        """
        tools = [ResponsesTool(type="function", name="shell",
                               description="run", parameters={"type": "object"})]
        unified = convert_responses_tools_to_unified(tools)
        assert len(unified) == 1
        assert unified[0].name == "shell"
        assert unified[0].input_schema == {"type": "object"}

    def test_non_function_tools_skipped(self):
        """
        What it does: built-in tools (web_search etc.) are skipped.
        Purpose: Kiro only accepts custom function tools.
        """
        tools = [ResponsesTool(type="web_search"),
                 ResponsesTool(type="function", name="f")]
        unified = convert_responses_tools_to_unified(tools)
        assert len(unified) == 1
        assert unified[0].name == "f"

    def test_empty_returns_none(self):
        assert convert_responses_tools_to_unified(None) is None
        assert convert_responses_tools_to_unified([]) is None


# ==================================================================================================
# extract_thinking_config_from_responses
# ==================================================================================================

class TestThinkingConfig:
    def test_no_reasoning_uses_default(self):
        req = ResponsesRequest(model="m", input="x")
        cfg = extract_thinking_config_from_responses(req)
        assert cfg.enabled is True
        assert cfg.budget_tokens is None

    def test_effort_none_disables(self):
        req = ResponsesRequest(model="m", input="x", reasoning={"effort": "none"})
        cfg = extract_thinking_config_from_responses(req)
        assert cfg.enabled is False

    def test_effort_high_sets_budget(self):
        req = ResponsesRequest(model="m", input="x",
                               reasoning={"effort": "high"}, max_output_tokens=4096)
        cfg = extract_thinking_config_from_responses(req)
        assert cfg.enabled is True
        assert cfg.budget_tokens == int(4096 * 0.8)

    def test_unknown_effort_falls_back_to_default(self):
        req = ResponsesRequest(model="m", input="x", reasoning={"effort": "bogus"})
        cfg = extract_thinking_config_from_responses(req)
        assert cfg.enabled is True
        assert cfg.budget_tokens is None


# ==================================================================================================
# build_kiro_payload (end to end)
# ==================================================================================================

class TestBuildKiroPayload:
    def test_basic_payload_structure(self):
        """
        What it does: builds a valid Kiro payload from a simple request.
        Purpose: end-to-end sanity of the Responses adapter.
        """
        req = ResponsesRequest(model="claude-sonnet-4.5",
                               instructions="be brief", input="hello")
        payload = build_kiro_payload(req, "conv-1", "arn:test")

        assert "conversationState" in payload
        cs = payload["conversationState"]
        assert cs["conversationId"] == "conv-1"
        assert payload["profileArn"] == "arn:test"
        current = cs["currentMessage"]["userInputMessage"]
        # system prompt is folded into the current message when history is empty
        assert "be brief" in current["content"]

    def test_tool_round_trip_payload(self):
        """
        What it does: a full call/result round trip produces tools + toolResults.
        Purpose: verify tool calls survive conversion into Kiro format.
        """
        req = ResponsesRequest(model="m", input=[
            {"type": "message", "role": "user",
             "content": [{"type": "input_text", "text": "run ls"}]},
            {"type": "function_call", "call_id": "c1", "name": "shell",
             "arguments": '{"cmd":"ls"}'},
            {"type": "function_call_output", "call_id": "c1", "output": "a.txt"},
        ], tools=[ResponsesTool(type="function", name="shell",
                                parameters={"type": "object"})])
        payload = build_kiro_payload(req, "conv-1", "arn:test")

        body = json.dumps(payload)
        assert "shell" in body
        assert "toolResults" in body or "toolUses" in body
