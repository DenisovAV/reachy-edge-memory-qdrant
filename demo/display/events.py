"""Server-Sent Events encoding: event name + JSON payload on one line.

A pure function, tested without sockets. SSE format: 'event: <name>\n
data: <json>\n\n' — the browser's EventSource parses this on its own.
"""
from __future__ import annotations

import json


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
