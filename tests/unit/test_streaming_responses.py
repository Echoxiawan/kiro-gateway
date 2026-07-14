# -*- coding: utf-8 -*-

"""
Unit tests for streaming_responses module (OpenAI Responses API SSE).

Tests the KiroEvent -> Responses SSE state machine by feeding a fake Kiro
event stream through the internal generator and parsing the emitted events.
"""

import json

import pytest

from kiro.streaming_core import KiroEvent
from kiro import streaming_responses


class _FakeResponse:
    """Minimal stand-in for httpx.Response.aclose()."""
    async def aclose(self):
        return None


def _parse_sse(chunks):
    """Parse a list of SSE strings into [(event_type, data_dict), ...]."""
    events = []
    for chunk in chunks:
        lines = chunk.strip().split("\n")
        event_type = None
        data = None
        for line in lines:
            if line.startswith("event: "):
                event_type = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        if event_type is not None:
            events.append((event_type, data))
    return events


async def _run(events, monkeypatch, model="claude-sonnet-4.5"):
    """Drive the internal generator with a fake parse_kiro_stream."""
    async def fake_parse_kiro_stream(response, first_token_timeout, *a, **kw):
        for e in events:
            yield e

    monkeypatch.setattr(streaming_responses, "parse_kiro_stream", fake_parse_kiro_stream)

    class _Cache:
        def get_max_input_tokens(self, model):
            return 200000

    chunks = []
    async for chunk in streaming_responses.stream_kiro_to_responses_internal(
        client=None, response=_FakeResponse(), model=model,
        model_cache=_Cache(), auth_manager=None,
    ):
        chunks.append(chunk)
    return _parse_sse(chunks)


@pytest.mark.asyncio
class TestResponsesStreaming:

    async def test_text_only_event_sequence(self, monkeypatch):
        """
        What it does: content events produce a well-formed message item lifecycle.
        Purpose: Codex expects created -> item.added -> deltas -> done -> completed.
        """
        events = [
            KiroEvent(type="content", content="Hello "),
            KiroEvent(type="content", content="world"),
            KiroEvent(type="context_usage", context_usage_percentage=1.0),
        ]
        parsed = await _run(events, monkeypatch)
        types = [t for t, _ in parsed]

        assert types[0] == "response.created"
        assert "response.output_item.added" in types
        assert "response.content_part.added" in types
        assert types.count("response.output_text.delta") == 2
        assert "response.output_text.done" in types
        assert "response.output_item.done" in types
        assert types[-1] == "response.completed"

    async def test_deltas_carry_text(self, monkeypatch):
        """
        What it does: output_text.delta events carry the exact content chunks.
        """
        events = [KiroEvent(type="content", content="abc")]
        parsed = await _run(events, monkeypatch)
        deltas = [d["delta"] for t, d in parsed if t == "response.output_text.delta"]
        assert deltas == ["abc"]

    async def test_completed_has_usage_and_output(self, monkeypatch):
        """
        What it does: response.completed carries usage (input/output tokens) and output items.
        Purpose: Codex reads final usage and output from this event.
        """
        events = [
            KiroEvent(type="content", content="hi"),
            KiroEvent(type="context_usage", context_usage_percentage=2.0),
        ]
        parsed = await _run(events, monkeypatch)
        _, completed = [e for e in parsed if e[0] == "response.completed"][0]
        assert "usage" in completed["response"]
        assert "input_tokens" in completed["response"]["usage"]
        assert "output_tokens" in completed["response"]["usage"]
        assert completed["response"]["status"] == "completed"
        assert len(completed["response"]["output"]) == 1

    async def test_reasoning_then_text(self, monkeypatch):
        """
        What it does: thinking events open a reasoning item, then content opens a message.
        Purpose: reasoning and message are distinct output items.
        """
        events = [
            KiroEvent(type="thinking", thinking_content="let me think"),
            KiroEvent(type="content", content="answer"),
        ]
        parsed = await _run(events, monkeypatch)
        types = [t for t, _ in parsed]

        assert "response.reasoning_summary_text.delta" in types
        assert "response.output_text.delta" in types
        # reasoning item closes before the message item opens
        assert types.index("response.reasoning_summary_text.done") < \
               types.index("response.output_text.delta")

    async def test_function_call_events(self, monkeypatch):
        """
        What it does: a tool_use event produces a function_call item with args.
        Purpose: Codex needs function_call items to run tools.
        """
        events = [
            KiroEvent(type="content", content="calling"),
            KiroEvent(type="tool_use", tool_use={
                "id": "call_1", "type": "function",
                "function": {"name": "shell", "arguments": '{"cmd":"ls"}'},
            }),
        ]
        parsed = await _run(events, monkeypatch)
        types = [t for t, _ in parsed]

        assert "response.function_call_arguments.delta" in types
        assert "response.function_call_arguments.done" in types

        # the completed response includes a function_call output item
        _, completed = [e for e in parsed if e[0] == "response.completed"][0]
        fc = [i for i in completed["response"]["output"] if i["type"] == "function_call"]
        assert len(fc) == 1
        assert fc[0]["call_id"] == "call_1"
        assert fc[0]["name"] == "shell"
        assert fc[0]["arguments"] == '{"cmd":"ls"}'

    async def test_sequence_numbers_monotonic(self, monkeypatch):
        """
        What it does: every event carries a strictly increasing sequence_number.
        Purpose: Codex relies on ordered sequence numbers.
        """
        events = [
            KiroEvent(type="thinking", thinking_content="t"),
            KiroEvent(type="content", content="c"),
        ]
        parsed = await _run(events, monkeypatch)
        seqs = [d["sequence_number"] for _, d in parsed]
        assert seqs == list(range(len(seqs)))
