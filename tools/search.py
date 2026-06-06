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
    "today", "current", "recent", "headlines",
    "world",
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
            answer = _search_duckduckgo(query, max_results)
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


def _search_duckduckgo(query: str, max_results: int) -> str:
    ddgs = DDGS()
    candidate_queries = _candidate_queries(query)
    attempts = []

    if _is_news_query(query):
        for candidate in candidate_queries:
            attempts.extend([
                lambda candidate=candidate: ddgs.news(candidate, max_results=max_results, timelimit="d"),
                lambda candidate=candidate: ddgs.news(candidate, max_results=max_results, timelimit="w"),
            ])

    for candidate in candidate_queries:
        attempts.append(lambda candidate=candidate: ddgs.text(candidate, max_results=max_results))

    errors = []
    for attempt in attempts:
        try:
            answer = _combine_snippets(attempt())
            _validate_answer(query, answer)
            return answer
        except ToolError as exc:
            errors.append(str(exc))

    reason = "; ".join(errors) if errors else "no search results"
    raise ToolError(reason, tool_name="search")


def _candidate_queries(query: str) -> list[str]:
    normalized = _normalize_query(query)
    candidates = [normalized]

    if _is_news_query(normalized):
        topic = _freshness_topic(normalized)
        if topic:
            candidates = [
                f"latest {topic} news",
                f"{topic} latest updates",
                *candidates,
            ]
        elif "world" in normalized:
            candidates = ["latest world news", *candidates]

    return list(dict.fromkeys(candidate for candidate in candidates if candidate))


def _combine_snippets(results: Any) -> str:
    snippets = []
    for r in list(results)[:MAX_RESULTS]:
        title   = str(r.get("title")   or "").strip()
        snippet = str(r.get("body")    or r.get("snippet") or "").strip()
        href    = str(r.get("href")    or r.get("url") or "").strip()
        source  = str(r.get("source")  or "").strip()
        date    = str(r.get("date")    or "").strip()

        if title and snippet:
            snippets.append(f"{title}: {snippet}")
        elif title:
            snippets.append(title)
        elif snippet:
            snippets.append(snippet)

        meta = " · ".join(part for part in [source, date] if part)
        if meta:
            snippets.append(f"Meta: {meta}")

        if href:
            snippets.append(f"Source: {href}")

    return "\n".join(snippets).strip()


def _validate_answer(query: str, answer: str) -> None:
    if not answer:
        raise ToolError("no search results", tool_name="search")

    min_length = 24 if _is_news_query(query) else MIN_RESULT_LENGTH
    if len(answer) < min_length:
        raise ToolError("result too short", tool_name="search")

    required_tokens = _meaningful_tokens(query)
    if required_tokens and not any(token in answer.lower() for token in required_tokens):
        raise ToolError("result unrelated to query", tool_name="search")


def _is_news_query(query: str) -> bool:
    lower = query.lower()
    return any(
        token in lower
        for token in (
            "news", "latest", "today", "headlines", "current events",
            "what's new", "what is new", "new in", "recent", "current",
        )
    )


def _freshness_topic(query: str) -> str:
    lower = _normalize_query(query)
    patterns = [
        r"what'?s\s+new\s+(?:in|with|about)\s+(.+)",
        r"what\s+is\s+new\s+(?:in|with|about)\s+(.+)",
        r"new\s+(?:in|with|about)\s+(.+)",
        r"latest\s+(.+?)(?:\s+today)?$",
        r"recent\s+(.+?)(?:\s+news|updates)?$",
        r"current\s+(.+?)(?:\s+news|updates)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, lower)
        if match:
            topic = _clean_topic(match.group(1))
            if topic:
                return topic
    return ""


def _clean_topic(topic: str) -> str:
    topic = re.sub(r"\b(world|today|news|headlines|updates|current|recent|latest)\b", " ", topic)
    topic = re.sub(r"\s+", " ", topic).strip(" ?.,")
    return topic


def _normalize_query(query: str) -> str:
    return " ".join(query.lower().replace("’", "'").split())


def _meaningful_tokens(query: str) -> set[str]:
    tokens = set(re.findall(r"\b[a-z0-9]{4,}\b", query.lower()))
    return {t for t in tokens if t not in _STOP_WORDS}
