"""Shared test fixtures."""

from __future__ import annotations

import re
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # allow `pytest` from a bare checkout
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel  # noqa: E402
from langchain_core.messages import AIMessage, AIMessageChunk  # noqa: E402
from langchain_core.outputs import ChatGenerationChunk, ChatResult  # noqa: E402


class _Replay:
    """Scripted conversation that survives a `raise`.

    A generator dies as soon as an item raises, so the harness's turn retry
    would call `next()` on a dead script and hit `StopIteration` (which
    langgraph turns into `RuntimeError: generator raised StopIteration`).
    An iterator object just carries on from the next item instead.
    """

    def __init__(self, items: Sequence[AIMessage | BaseException]) -> None:
        self._items = iter(items)

    def __iter__(self) -> "_Replay":
        return self

    def __next__(self) -> AIMessage:
        item = next(self._items)
        if isinstance(item, BaseException):
            raise item  # scripted model-call failure (flaky endpoint)
        return item  # type: ignore[return-value]


class FakeToolModel(GenericFakeChatModel):
    """`GenericFakeChatModel` that tolerates `bind_tools` and streams tool calls.

    The stock fake model emits nothing for messages that only carry
    `tool_calls`, which breaks streaming (`No generation chunks were returned`),
    and it does not implement `bind_tools`, which deepagents requires.
    """

    def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003, ARG002
        return self

    def _stream(self, messages, stop=None, run_manager=None, **kwargs):  # noqa: ANN001, ANN003
        result: ChatResult = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        message = result.generations[0].message
        content = message.content if isinstance(message.content, str) else ""
        pieces = re.split(r"(\s)", content) if content else [""]
        for index, piece in enumerate(pieces):
            last = index == len(pieces) - 1
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=piece,
                    id=message.id,
                    tool_calls=message.tool_calls if last else [],
                    additional_kwargs=message.additional_kwargs if last else {},
                )
            )


def ai(text: str = "") -> AIMessage:
    return AIMessage(content=text)


def tool_call(name: str, args: dict, call_id: str) -> AIMessage:
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


@pytest.fixture
def fake_model():
    """Build a fake model that replays a fixed sequence of messages."""

    def _make(messages: Sequence[AIMessage | BaseException], sleep: float | None = None) -> FakeToolModel:
        kwargs: dict = {"messages": _Replay(messages)}
        if sleep is not None:
            kwargs["sleep"] = sleep
        return FakeToolModel(**kwargs)

    return _make
