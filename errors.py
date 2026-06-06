from __future__ import annotations


class ToolError(Exception):
    def __init__(self, message: str, tool_name: str | None = None) -> None:
        super().__init__(message)
        self.tool_name = tool_name