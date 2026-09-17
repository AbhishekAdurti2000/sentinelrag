"""Thin LLM boundary.

Everything in the orchestrator talks to `LLMClient`, never to the Anthropic
SDK directly. That means the entire agent loop, guardrails, and tool
dispatch can be unit-tested and run in CI with `FakeLLMClient` and zero API
key / network access -- only the small `AnthropicClient` adapter itself is
untested-by-CI (it's exercised in the live end-to-end demo instead). This
split is exactly what makes "eval + guardrails in CI" tractable without
burning API credits on every push.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class TextBlock:
    text: str
    type: str = "text"


@dataclass
class ToolUseBlock:
    id: str
    name: str
    input: dict[str, Any]
    type: str = "tool_use"


@dataclass
class LLMResponse:
    content: list[TextBlock | ToolUseBlock]
    stop_reason: str  # "end_turn" | "tool_use" | "max_tokens"

    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]


class LLMClient(Protocol):
    def call(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse: ...


class AnthropicClient:
    """Real client, backed by the Anthropic Messages API."""

    def __init__(self, api_key: str, model: str):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def call(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = dict(
            model=self.model,
            max_tokens=1500,
            system=system,
            messages=messages,
        )
        if tools:
            kwargs["tools"] = tools
        resp = self._client.messages.create(**kwargs)

        blocks: list[TextBlock | ToolUseBlock] = []
        for block in resp.content:
            if block.type == "text":
                blocks.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                blocks.append(ToolUseBlock(id=block.id, name=block.name, input=block.input))
        return LLMResponse(content=blocks, stop_reason=resp.stop_reason)


@dataclass
class FakeLLMClient:
    """Deterministic stand-in for tests/CI. `script` is a list of LLMResponse
    objects returned in order, one per call() invocation -- lets a test
    script an exact planner -> tool_use -> tool_use -> final_answer sequence.
    """

    script: list[LLMResponse]
    calls: list[dict[str, Any]] = field(default_factory=list)

    def call(
        self,
        messages: list[dict[str, Any]],
        system: str,
        tools: list[dict[str, Any]] | None = None,
    ) -> LLMResponse:
        self.calls.append({"messages": messages, "system": system, "tools": tools})
        idx = len(self.calls) - 1
        if idx >= len(self.script):
            raise AssertionError("FakeLLMClient script exhausted -- add more scripted responses")
        return self.script[idx]
