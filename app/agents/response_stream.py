"""Separate public action updates from answers without exposing provider reasoning.

Tagged output can stream immediately. Untagged output waits for the completed
AI message, since text tokens can precede a tool call in the same model turn.
"""
from __future__ import annotations

from typing import Any, Callable


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            block if isinstance(block, str) else block.get("text", "")
            for block in content
            if isinstance(block, str)
            or (isinstance(block, dict) and block.get("type") in {"text", "output_text"})
        )
    return ""


class ResponseStream:
    """One model turn, including delimiters split across arbitrary token boundaries."""

    def __init__(self, on_progress: Callable, on_answer: Callable) -> None:
        self.on_progress = on_progress
        self.on_answer = on_answer
        self.raw = ""
        self.pending = ""
        self.channel = ""
        self.answer = ""
        self.progress = ""
        self.finished = False

    def _write(self, text: str) -> None:
        if not text:
            return
        if self.channel == "progress":
            self.progress += text
            self.on_progress(text, False)
        elif self.channel == "answer":
            self.answer += text
            self.on_answer(text)

    def feed(self, text: str) -> None:
        if self.finished:
            return
        self.raw += text
        self.pending += text
        while self.pending:
            if not self.channel:
                self.pending = self.pending.lstrip()
                tags = {"<progress>": "progress", "<answer>": "answer"}
                for tag, channel in tags.items():
                    if self.pending.startswith(tag):
                        self.channel = channel
                        self.pending = self.pending[len(tag):]
                        break
                else:
                    if any(tag.startswith(self.pending) for tag in tags):
                        return
                    # No declared channel: classify only once tool_calls is known.
                    self.channel = "unclassified"
                    return
            if self.channel == "unclassified":
                return
            closing = f"</{self.channel}>"
            index = self.pending.find(closing)
            if index >= 0:
                self._write(self.pending[:index])
                self.pending = self.pending[index + len(closing):]
                self.channel = ""
                continue
            # Keep only a possible closing-delimiter prefix, never its raw tokens.
            keep = 0
            for size in range(1, min(len(closing), len(self.pending) + 1)):
                if self.pending.endswith(closing[:size]):
                    keep = size
            text = self.pending[:-keep] if keep else self.pending
            self.pending = self.pending[-keep:] if keep else ""
            self._write(text)
            return

    def finish(self, *, tool_calls: bool = False) -> str:
        if self.finished:
            return self.answer.strip()
        if self.channel == "unclassified" or not self.channel:
            self.channel = "progress" if tool_calls else "answer"
        self._write(self.pending)
        self.pending = ""
        if self.progress:
            self.on_progress("", True)
        self.finished = True
        return "" if tool_calls else self.answer.strip()


class ProgressEventCollector:
    """SSE event order and compact history share a single sequence counter."""

    def __init__(self) -> None:
        self.sequence = 0
        self.steps: list[dict] = []
        self.artifacts: list[dict] = []
        self._streams: dict[str, dict] = {}
        self.answer_streamed = False

    def _ordered(self, event: dict) -> dict:
        self.sequence += 1
        return {**event, "sequence": self.sequence}

    def step(self, step: str, detail: str) -> dict:
        event = self._ordered({"step": step, "detail": detail})
        self.steps.append(dict(event))
        return {"type": "status", **event}

    def artifact(self, artifact: dict) -> dict:
        event = self._ordered(artifact)
        if event.get("kind") == "answer":
            self.answer_streamed |= bool(event.get("content"))
        elif event.get("id") and "streaming" in event:
            stream_id = event["id"]
            saved = self._streams.get(stream_id)
            if saved is None:
                saved = {**event, "streaming": False}
                self._streams[stream_id] = saved
                self.artifacts.append(saved)
            else:
                saved["content"] += event.get("content", "")
        else:
            self.artifacts.append(dict(event))
        return {"type": "artifact", **event}
