from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Literal

from dotenv import load_dotenv
from groq import Groq

from errors import ToolError
from logger import log_error, log_fallback, log_request
from tools import calculator, search


load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Groq client instantiated once at module level for efficiency
_groq_client: Groq | None = None


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise AgentError("GROQ_API_KEY is not configured")
        _groq_client = Groq(api_key=api_key)
    return _groq_client


DIRECT_SYSTEM_PROMPT = (
    "You are a helpful AI assistant with access to a calculator and web search. "
    "Answer clearly and concisely. If a tool failed, acknowledge it and answer "
    "from your own knowledge."
)

Intent    = Literal["calc", "search", "direct"]
Route     = Literal[
    "CALCULATOR",
    "SEARCH",
    "FALLBACK_FROM_CALC",
    "FALLBACK_FROM_SEARCH",
    "DIRECT_LLM",
]
Confidence = Literal["high", "medium", "low"]


class AgentError(RuntimeError):
    pass


@dataclass
class LLMResult:
    answer:     str
    confidence: Confidence


@dataclass
class AgentResponse:
    answer:         str
    route:          Route
    tool_attempted: Literal["calculator", "search", "none"]
    fallback_reason: str | None
    confidence:     Confidence

    def to_dict(self) -> dict[str, str | None]:
        return {
            "answer":          self.answer,
            "route":           self.route,
            "tool_attempted":  self.tool_attempted,
            "fallback_reason": self.fallback_reason,
            "confidence":      self.confidence,
        }


class Agent:
    def answer(self, query: str) -> dict[str, str | None]:
        clean_query = self._validate_query(query)
        intent      = classify_intent(clean_query)

        if intent == "calc":
            response = self._answer_with_calculator(clean_query)
        elif intent == "search":
            response = self._answer_with_search(clean_query)
        else:
            response = self._answer_directly(clean_query)

        return response.to_dict()

    def _answer_with_calculator(self, query: str) -> AgentResponse:
        try:
            result = calculator.run(query)
            answer = f"The result is {result}"
            response = AgentResponse(answer, "CALCULATOR", "calculator", None, "high")
            log_request(query, response.route, answer=answer)
            return response
        except ToolError as exc:
            reason = str(exc)
            log_fallback(query, "calculator", reason=reason)
            llm_result = groq_llm_answer(
                query,
                system_prompt=_fallback_system_prompt(reason),
                context=f"Tool calculator failed: {reason}",
            )
            response = AgentResponse(
                llm_result.answer,
                "FALLBACK_FROM_CALC",
                "calculator",
                reason,
                llm_result.confidence,
            )
            log_request(query, response.route, answer=llm_result.answer)
            return response

    def _answer_with_search(self, query: str) -> AgentResponse:
        try:
            answer = search.run(query)
            response = AgentResponse(answer, "SEARCH", "search", None, "high")
            log_request(query, response.route, answer=answer)
            return response
        except ToolError as exc:
            reason = str(exc)
            log_fallback(query, "search", reason=reason)
            llm_result = groq_llm_answer(
                query,
                system_prompt=_fallback_system_prompt(reason),
                context=f"Tool search failed: {reason}",
            )
            response = AgentResponse(
                llm_result.answer,
                "FALLBACK_FROM_SEARCH",
                "search",
                reason,
                llm_result.confidence,
            )
            log_request(query, response.route, answer=llm_result.answer)
            return response

    def _answer_directly(self, query: str) -> AgentResponse:
        llm_result = groq_llm_answer(query, system_prompt=DIRECT_SYSTEM_PROMPT)
        response   = AgentResponse(
            llm_result.answer, "DIRECT_LLM", "none", None, llm_result.confidence
        )
        log_request(query, response.route, answer=response.answer)
        return response

    @staticmethod
    def _validate_query(query: str) -> str:
        clean = query.strip()
        if not clean:
            raise AgentError("Query must not be empty")
        return clean


def classify_intent(query: str) -> Intent:
    lower = query.lower().strip()

    # --- direct: identity / greetings ---
    direct_triggers = [
        "who are you", "what are you", "what can you do",
        "your name",
    ]
    if lower in {"hello", "hi", "hey", "help"}:
        return "direct"
    if any(t in lower for t in direct_triggers):
        return "direct"

    # --- calc: trust looks_like_math first, then keywords ---
    if calculator.looks_like_math(lower):
        return "calc"

    calc_keywords = ["calculate", "compute", "eval", "evaluate", "sqrt"]
    if any(kw in lower for kw in calc_keywords):
        return "calc"

    if re.search(r"\bwhat\s+is\s+\d", lower):
        return "calc"

    if needs_search(lower):
        return "search"

    return "direct"


def needs_search(query: str) -> bool:
    lower = query.lower().strip()

    freshness_keywords = [
        "latest", "recent", "current", "today", "yesterday",
        "this week", "this month", "right now", "breaking",
        "news", "headlines", "trending", "updates", "what's new",
        "what is new", "new in", "new with",
    ]
    if any(keyword in lower for keyword in freshness_keywords):
        return True

    # --- search: everything else that sounds like a lookup ---
    search_keywords = [
        "search", "find", "look up", "who is", "what is",
        "tell me about", "news", "latest", "recent",
        "current events", "headlines", "explain",
        "history", "where is", "describe",
    ]
    if any(kw in lower for kw in search_keywords):
        return True

    lookup_question = (
        r"^(who|what|where|when|which)\b|"
        r"^how\s+(many|much|old|far|long|tall|big|large|small|fast)\b"
    )
    if re.search(lookup_question, lower):
        return True

    factual_entities = [
        "price", "stock", "weather", "score", "schedule",
        "release date", "version", "ceo", "president", "capital",
        "population", "definition", "meaning",
    ]
    return any(keyword in lower for keyword in factual_entities)


def _fallback_system_prompt(reason: str) -> str:
    return (
        f"A tool was attempted but failed with reason: {reason}.\n"
        "Answer the user's question from your own knowledge."
    )


def groq_llm_answer(
    user_query:    str,
    system_prompt: str = DIRECT_SYSTEM_PROMPT,
    context:       str | None = None,
) -> LLMResult:
    try:
        client       = _get_groq_client()
        user_content = f"{context}\n\n{user_query}" if context else user_query
        response     = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_content},
            ],
            max_tokens=512,
        )
        answer = response.choices[0].message.content
        if not answer:
            raise AgentError("Groq returned an empty response")
        confidence: Confidence = "medium" if context else "high"
        return LLMResult(answer=answer.strip(), confidence=confidence)

    except Exception as exc:
        reason = str(exc)
        log_error(user_query, reason=reason)
        return LLMResult(
            answer=_local_llm_fallback(user_query, context, reason),
            confidence="low",
        )


def _local_llm_fallback(user_query: str, context: str | None, reason: str) -> str:
    lower = user_query.lower().strip()
    reason_lower = reason.lower()

    if "who are you" in lower or "what are you" in lower:
        return (
            "I am an AI assistant with calculator, "
            "web search, and Groq-backed direct answer routes."
        )
    if "decommissioned" in reason_lower or "model" in reason_lower:
        return (
            "The configured Groq model is unavailable. "
            f"Set GROQ_MODEL to a supported model such as {MODEL}, then restart the app."
        )
    if context:
        return (
            "A tool was attempted but failed, and I could not reach "
            "the Groq LLM. Please try again when network access is available."
        )
    return (
        "I could not reach the Groq LLM from this runtime. "
        "Please try again when network access is available."
    )
