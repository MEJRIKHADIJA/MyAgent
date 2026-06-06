from __future__ import annotations

from datetime import datetime
from pathlib import Path

from colorama import Fore, Style, init


init(autoreset=True)

LOG_FILE    = Path("agent_log.txt")
_MAX_QUERY  = 60
_MAX_DETAIL = 120


def log_request(query: str, route: str, answer: str) -> None:
    _write_log(route, query, f"answer: {_clean(answer)}")


def log_fallback(query: str, tool_name: str, reason: str) -> None:
    _write_log("FALLBACK", query, f"reason: {tool_name} failed: {_clean(reason)}")


def log_error(query: str, reason: str) -> None:
    _write_log("ERROR", query, f"reason: {_clean(reason)}")


def get_recent_logs(n: int = 20) -> list[dict[str, str]]:
    if not LOG_FILE.exists():
        return []
    lines = LOG_FILE.read_text(encoding="utf-8").splitlines()[-n:]
    return [_parse_log_line(line) for line in lines if line.strip()]


def _write_log(route: str, query: str, detail: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    short_query  = _truncate(_clean(query),  _MAX_QUERY)
    short_detail = _truncate(_clean(detail), _MAX_DETAIL)
    line = f'{timestamp} | {route:<22} | "{short_query}" | {short_detail}'

    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")

    print(_color_for_route(route) + line + Style.RESET_ALL)


def _parse_log_line(line: str) -> dict[str, str]:
    parts = [p.strip() for p in line.split("|", maxsplit=3)]
    if len(parts) != 4:
        return {"raw": line}
    timestamp, route, query, detail = parts
    query = query.strip('"')
    key, _, value = detail.partition(":")
    return {
        "timestamp": timestamp,
        "route":     route,
        "query":     query,
        (key.strip() or "detail"): value.strip(),
    }


def _clean(value: str) -> str:
    return " ".join(str(value).split())


def _truncate(value: str, max_len: int) -> str:
    return value if len(value) <= max_len else value[:max_len - 3] + "..."


def _color_for_route(route: str) -> str:
    if route in {"CALCULATOR", "SEARCH"}:
        return Fore.GREEN
    if "FALLBACK" in route:
        return Fore.YELLOW
    if route == "DIRECT_LLM":
        return Fore.CYAN
    if route == "ERROR":
        return Fore.RED
    return ""