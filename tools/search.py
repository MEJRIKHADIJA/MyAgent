from __future__ import annotations

import queue
import re
import threading
from typing import Any

from duckduckgo_search import DDGS

from errors import ToolError


_STOP_WORDS = {
    "about", "find", "for", "history", "latest",
    "look", "news", "search", "tell", "the",
    "what", "where", "who", "with", "me",
}

MIN_RESULT_LENGTH = 40
MAX_RESULTS       = 3
TIMEOUT_SECONDS   = 8


def run(query: str, max_results: int = MAX_RESULTS, timeout_seconds: int = TIMEOUT_SECONDS) -> str:
    result_queue: queue.Queue[str | ToolError] = queue.Queue(maxsize=1)

    def put_once(value: str | ToolError) -> None:
        try:
            result_queue.put_nowait(value)
        except queue.Full:
            pass

    def on_timeout() -> None:
        put_once(ToolError("timeout", tool_name="search"))

    def worker() -> None:
        try:
            results = DDGS().text(query, max_results=max_results)
            answer  = _combine_snippets(results)
            _validate_answer(query, answer)
            put_once(answer)
        except ToolError as exc:
            put_once(exc)
        except Exception as exc:
            put_once(ToolError(str(exc), tool_name="search"))

    timer  = threading.Timer(timeout_seconds, on_timeout)
    thread = threading.Thread(target=worker, daemon=True)
    timer.daemon = True

    timer.start()
    thread.start()
    result = result_queue.get()
    timer.cancel()

    if isinstance(result, ToolError):
        raise result
    return result


def _combine_snippets(results: Any) -> str:
    snippets = []
    for r in list(results)[:MAX_RESULTS]:
        title   = str(r.get("title")   or "").strip()
        snippet = str(r.get("body")    or r.get("snippet") or "").strip()
        href    = str(r.get("href")    or "").strip()

        if title and snippet:
            snippets.append(f"{title}: {snippet}")
        elif snippet:
            snippets.append(snippet)

        if href:
            snippets.append(f"Source: {href}")

    return "\n".join(snippets).strip()


def _validate_answer(query: str, answer: str) -> None:
    if not answer or len(answer) < MIN_RESULT_LENGTH:
        raise ToolError("result too short", tool_name="search")

    required_tokens = _meaningful_tokens(query)
    if required_tokens and not any(token in answer.lower() for token in required_tokens):
        raise ToolError("result unrelated to query", tool_name="search")


def _meaningful_tokens(query: str) -> set[str]:
    tokens = set(re.findall(r"\b[a-z0-9]{4,}\b", query.lower()))
    return {t for t in tokens if t not in _STOP_WORDS}