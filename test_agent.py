from __future__ import annotations

import httpx

BASE_URL = "http://localhost:8000"

TEST_CASES = [
    ("What is sqrt(144) + 5?",          ("CALCULATOR",)),
    ("What is 12 * 8?",                 ("CALCULATOR",)),
    ("Calculate (99 + 1) / 5",          ("CALCULATOR",)),
    ("Search for latest AI news",       ("SEARCH", "FALLBACK")),
    ("Tell me about Tunisia",           ("SEARCH", "FALLBACK", "DIRECT_LLM")),
    ("Who are you?",                    ("DIRECT_LLM",)),
    ("What is the capital of France?",  ("__NOT_EMPTY__",)),
    ("Search 404 xzqjfake notreal",     ("FALLBACK", "SEARCH")),
]


def main() -> None:
    passed = 0

    with httpx.Client(timeout=30.0, trust_env=False) as client:
        for i, (query, expected) in enumerate(TEST_CASES, start=1):
            try:
                resp = client.post(f"{BASE_URL}/ask", json={"query": query})
                resp.raise_for_status()
                data  = resp.json()
                route = str(data.get("route") or "")
                ok    = _route_matches(route, expected)
            except Exception as exc:
                route = f"ERROR ({exc})"
                ok    = False

            status = "PASS" if ok else "FAIL"
            print(f"{status} {i}: route={route!r:35} | {query}")
            if ok:
                passed += 1

    print(f"\nResult: {passed}/{len(TEST_CASES)} tests passed.")


def _route_matches(route: str, expected: tuple[str, ...]) -> bool:
    if "__NOT_EMPTY__" in expected:
        return bool(route)
    return any(e in route for e in expected)


if __name__ == "__main__":
    main()