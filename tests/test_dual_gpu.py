"""The dual Ollama runner must use both workers without losing responses."""

# ruff: noqa: S101 - assertions are the checks in this test module

from __future__ import annotations

import json
import hashlib
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from graphify.dual_gpu import Target, TargetPool, _source_snapshot, make_handler, verify_run


def _server(
    model_name: str, barrier: threading.Barrier, delay: float = 0
) -> tuple[ThreadingHTTPServer, threading.Thread]:
    first_request = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - http.server interface
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            assert payload["model"] == model_name
            if not first_request.is_set():
                first_request.set()
                barrier.wait(timeout=5)
            time.sleep(delay)
            body = json.dumps({"model": model_name, "choices": []}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread


def test_concurrent_chunks_reach_both_models(tmp_path: Path) -> None:
    """Two in-flight chunks use distinct upstreams and both return."""
    barrier = threading.Barrier(2)
    upstreams = [_server("model-0", barrier, delay=0.3), _server("model-1", barrier)]
    targets = [
        Target(f"http://127.0.0.1:{server.server_port}/v1", f"model-{i}")
        for i, (server, _) in enumerate(upstreams)
    ]
    pool = TargetPool(targets, run_id="run-123", journal_path=tmp_path / "dual-gpu-requests.jsonl")
    proxy = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(pool, 5))
    proxy_thread = threading.Thread(target=proxy.serve_forever, daemon=True)
    proxy_thread.start()

    def request() -> str:
        payload = json.dumps({"model": "dual-gpu", "messages": [], "stream": False}).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{proxy.server_port}/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=6) as response:  # noqa: S310 - loopback test server
            assert response.headers["X-Graphify-Run-ID"] == "run-123"
            assert response.headers["X-Graphify-Request-ID"]
            return json.load(response)["model"]

    try:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(request) for _ in range(3)]
            assert sorted(future.result(timeout=7) for future in futures) == [
                "model-0",
                "model-1",
                "model-1",
            ]
        assert pool.counts == [1, 2]
        assert pool.failures == [0, 0]
        journal = (tmp_path / "dual-gpu-requests.jsonl").read_text().splitlines()
        events = [json.loads(line) for line in journal]
        assert len(events) == 6
        assert {event["run_id"] for event in events} == {"run-123"}
        assert len({event["request_id"] for event in events}) == 3
        assert all("messages" not in event and "api_key" not in event for event in events)
        assert all(
            len(event["input_sha256"]) == 64 for event in events if event["phase"] == "started"
        )
        assert all(
            len(event["response_sha256"]) == 64 for event in events if event["phase"] == "completed"
        )
    finally:
        proxy.shutdown()
        proxy.server_close()
        proxy_thread.join(timeout=2)
        for server, thread in upstreams:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


def test_source_snapshot_detects_input_change(tmp_path: Path) -> None:
    """A changed file invalidates the run's source snapshot."""
    source = tmp_path / "source"
    source.mkdir()
    code = source / "example.py"
    code.write_text("value = 1\n")
    original, count = _source_snapshot(source)
    assert count == 1
    code.write_text("value = 2\n")
    changed, changed_count = _source_snapshot(source)
    assert changed_count == count
    assert changed != original


def test_journal_write_failure_is_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed durable journal write remains visible to the caller."""
    pool = TargetPool([], run_id="run-1", journal_path=tmp_path / "dual-gpu-requests.jsonl")

    def fail_sync(_fd: int) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(os, "fsync", fail_sync)
    with pytest.raises(OSError, match="disk full"):
        pool.journal({"phase": "started", "request_id": "request-1"})
    assert pool.journal_errors == ["disk full"]


def test_verify_run_detects_tampered_graph_and_journal(tmp_path: Path) -> None:
    """Post-run verification detects changes to the graph or journal."""
    output = tmp_path
    graph = output / "graphify-out" / "graph.json"
    graph.parent.mkdir()
    graph.write_text('{"nodes": []}')
    journal = output / "dual-gpu-requests.jsonl"
    journal.write_text(
        json.dumps({"run_id": "run-1", "phase": "started", "request_id": "req-1"})
        + "\n"
        + json.dumps({"run_id": "run-1", "phase": "completed", "request_id": "req-1"})
        + "\n"
    )
    status = {
        "run_id": "run-1",
        "requests_per_gpu": [1, 0],
        "graph_sha256": hashlib.sha256(graph.read_bytes()).hexdigest(),
        "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
    }
    (output / "dual-gpu-status.json").write_text(json.dumps(status))
    assert verify_run(output) == []
    graph.write_text('{"nodes": [1]}')
    assert "graph hash mismatch" in verify_run(output)
    graph.write_text('{"nodes": []}')
    journal.write_text(journal.read_text() + "corrupted\n")
    assert "request journal hash mismatch" in verify_run(output)
