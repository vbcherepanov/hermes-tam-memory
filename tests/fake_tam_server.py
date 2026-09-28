"""Scripted stand-in for ``tam`` (stdio MCP) used by the unit tests.

Behaviour is driven by environment variables so tests can run it as a subprocess:

- ``FAKE_TAM_STORE``  JSON file holding saved records (survives restarts, like TAM's DB)
- ``FAKE_TAM_LOG``    JSON-lines log of every tools/call received
- ``FAKE_TAM_BACKEND`` ``local`` (default) or ``team`` response shapes
- ``FAKE_TAM_REJECT`` substring that makes memory_save answer as a quality-gate rejection
- ``FAKE_TAM_EXIT_AFTER`` exit after this many tools/call requests (crash simulation)
- ``FAKE_TAM_RECALL_DELAY`` seconds to sleep before answering memory_recall
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

STORE = Path(os.environ["FAKE_TAM_STORE"])
LOG = Path(os.environ["FAKE_TAM_LOG"]) if os.environ.get("FAKE_TAM_LOG") else None
BACKEND = os.environ.get("FAKE_TAM_BACKEND", "local")
REJECT = os.environ.get("FAKE_TAM_REJECT", "")
EXIT_AFTER = int(os.environ.get("FAKE_TAM_EXIT_AFTER", "0"))
RECALL_DELAY = float(os.environ.get("FAKE_TAM_RECALL_DELAY", "0"))


def load() -> list[dict[str, Any]]:
    return json.loads(STORE.read_text()) if STORE.exists() else []


def save(records: list[dict[str, Any]]) -> None:
    STORE.write_text(json.dumps(records))


def tool_names() -> list[str]:
    names = ["memory_save", "memory_recall", "memory_stats"]
    return [*names, "memory_scopes"] if BACKEND == "team" else names


def memory_save(args: dict[str, Any]) -> dict[str, Any]:
    if REJECT and REJECT in args["content"]:
        if BACKEND == "team":
            return {"scope": {"kind": "personal"}, "data": {"saved": False, "quality": {"reason": "low value"}}}
        return {"saved": False, "rejected_by_quality_gate": True, "score": 0.2, "reason": "low value"}
    records = load()
    record = {
        "id": len(records) + 1,
        "content": args["content"],
        "type": args.get("type", "fact"),
        "project": args.get("project", "general"),
        "tags": args.get("tags", []),
        "context": args.get("context", ""),
        "created_at": "2026-09-25T10:00:00Z",
    }
    records.append(record)
    save(records)
    if BACKEND == "team":
        return {"scope": {"kind": "personal"}, "data": {**record, "saved": True, "deduplicated": False}}
    return {"saved": True, "id": record["id"], "deduplicated": False}


def memory_recall(args: dict[str, Any]) -> dict[str, Any]:
    time.sleep(RECALL_DELAY)
    words = {w.lower() for w in args["query"].split() if len(w) > 2}
    hits = []
    for record in load():
        if args.get("project") and record["project"] != args["project"]:
            continue
        overlap = len(words & {w.lower().strip(".,?!") for w in record["content"].split()})
        if overlap:
            hits.append({**record, "score": float(overlap)})
    hits.sort(key=lambda r: r["score"], reverse=True)
    hits = hits[: args.get("limit", 10)]
    if BACKEND == "team":
        return {
            "results": [
                {
                    "scope": {"kind": "personal"},
                    "record": {**h, "created_by": {"display_name": "Alice"}},
                    "scope_rank": i,
                }
                for i, h in enumerate(hits, 1)
            ],
            "ordering": "scope_rank_then_score",
        }
    grouped: dict[str, list[dict[str, Any]]] = {}
    for h in hits:
        grouped.setdefault(h.pop("type"), []).append(h)
    return {"query": args["query"], "total": len(hits), "results": grouped}


def reply(message_id: Any, result: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps({"jsonrpc": "2.0", "id": message_id, "result": result}) + "\n")
    sys.stdout.flush()


def main() -> None:
    calls = 0
    sys.stderr.write("fake tam starting\n")
    for line in sys.stdin:
        message = json.loads(line)
        method = message.get("method")
        if "id" not in message:
            continue
        if method == "initialize":
            reply(
                message["id"],
                {
                    "protocolVersion": message["params"]["protocolVersion"],
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "total-agent-memory", "version": "fake"},
                },
            )
        elif method == "tools/list":
            reply(message["id"], {"tools": [{"name": n, "inputSchema": {"type": "object"}} for n in tool_names()]})
        elif method == "tools/call":
            calls += 1
            name, args = message["params"]["name"], message["params"]["arguments"]
            if LOG:
                with LOG.open("a") as fh:
                    fh.write(json.dumps({"name": name, "arguments": args}) + "\n")
            if name == "memory_save":
                result = memory_save(args)
            elif name == "memory_recall":
                result = memory_recall(args)
            else:
                reply(message["id"], {"content": [{"type": "text", "text": "unsupported"}], "isError": True})
                continue
            reply(message["id"], {"content": [{"type": "text", "text": json.dumps(result)}], "isError": False})
            if EXIT_AFTER and calls >= EXIT_AFTER:
                sys.exit(3)
        else:
            sys.stdout.write(
                json.dumps({"jsonrpc": "2.0", "id": message["id"], "error": {"code": -32601, "message": "not found"}})
                + "\n"
            )
            sys.stdout.flush()


if __name__ == "__main__":
    main()
