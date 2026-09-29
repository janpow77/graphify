"""Run one Graphify extraction across two local Ollama instances.

Graphify already partitions semantic files into chunks and merges their results
in submission order. This module only assigns concurrent requests to the next
available Ollama instance, so both GPUs work on the same extraction run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit


_SEMANTIC_EXCLUDES = (
    "*.pdf",
    "*.png",
    "*.svg",
    "*.jpg",
    "*.jpeg",
    "*.webp",
    "*.gif",
    "*.bmp",
    "*.tif",
    "*.tiff",
    "*.ico",
    "*.html",
    "*.htm",
    "*.yaml",
    "*.yml",
    "*.mdx",
    "*.qmd",
    "*.skill",
    "*.rst",
    "*.docx",
    "*.pptx",
    "*.xlsx",
    "*.gdoc",
    "*.gsheet",
    "*.gslides",
    "graphify-out/**",
)


def _check_semantic_scope(source: Path) -> None:
    """Reject a corpus if any non-code semantic file escapes the exclusions."""

    from graphify.detect import detect

    with tempfile.TemporaryDirectory(prefix="graphify-scope-") as temporary:
        found = detect(
            source,
            cache_root=Path(temporary),
            extra_excludes=list(_SEMANTIC_EXCLUDES),
        ).get("files", {})
    unexpected = [
        str(path)
        for kind in ("document", "paper", "image")
        for name in found.get(kind, [])
        if (path := Path(name)).suffix.lower() not in (".md", ".txt")
    ]
    if unexpected:
        raise RuntimeError(f"semantic scope contains unsupported files: {unexpected[:5]}")


@dataclass(frozen=True)
class Target:
    """One Ollama endpoint and its local model name."""

    base_url: str
    model: str

    @property
    def api_url(self) -> str:
        """Return the native Ollama API root."""
        return self.base_url.removesuffix("/v1")


class TargetPool:
    """Give each in-flight request exclusive use of one GPU endpoint."""

    def __init__(
        self,
        targets: list[Target],
        *,
        run_id: str = "",
        journal_path: Path | None = None,
    ) -> None:
        """Register targets in command-line order."""
        self.targets = targets
        self.available: queue.Queue[int] = queue.Queue()
        self.counts = [0] * len(targets)
        self.failures = [0] * len(targets)
        self.lock = threading.Lock()
        self.run_id = run_id
        self.journal_path = journal_path
        self.journal_errors: list[str] = []
        if journal_path is not None:
            journal_path.write_bytes(b"")
        for index in range(len(targets)):
            self.available.put(index)

    def record(self, index: int, *, failed: bool) -> None:
        """Track completed requests for the end-of-run status report."""
        with self.lock:
            self.counts[index] += 1
            self.failures[index] += int(failed)

    def journal(self, event: dict) -> None:
        """Persist an event before acknowledging a response to Graphify."""
        if self.journal_path is None:
            return
        entry = {"run_id": self.run_id, **event}
        line = (json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with self.lock:
            try:
                with self.journal_path.open("ab") as stream:
                    stream.write(line)
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as exc:
                self.journal_errors.append(str(exc))
                raise


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_snapshot(source: Path) -> tuple[str, int]:
    """Hash the files Graphify's collector can read, plus scan configuration."""
    from graphify.extract import collect_files

    paths = set(collect_files(source))
    paths.update(p for name in (".graphifyignore", ".gitignore") if (p := source / name).is_file())
    digest = hashlib.sha256()
    for path in sorted(paths):
        name = path.relative_to(source).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(bytes.fromhex(_file_hash(path)))
    return digest.hexdigest(), len(paths)


def verify_run(output: Path) -> list[str]:
    """Check persisted request records and graph against the run status."""
    status = json.loads((output / "dual-gpu-status.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    journal = output / "dual-gpu-requests.jsonl"
    if not journal.is_file() or _file_hash(journal) != status.get("journal_sha256"):
        problems.append("request journal hash mismatch")
    else:
        started: set[str] = set()
        completed: set[str] = set()
        try:
            for line in journal.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                request_id = event["request_id"]
                if event["run_id"] != status["run_id"]:
                    problems.append("foreign run ID in request journal")
                if event["phase"] == "started":
                    if request_id in started:
                        problems.append("duplicate request ID")
                    started.add(request_id)
                elif event["phase"] == "completed":
                    if request_id in completed or request_id not in started:
                        problems.append("unpaired completion")
                    completed.add(request_id)
                else:
                    problems.append("unknown journal phase")
        except (KeyError, TypeError, ValueError):
            problems.append("malformed request journal")
        if started != completed or len(completed) != sum(status["requests_per_gpu"]):
            problems.append("incomplete request journal")
    graph = output / "graphify-out" / "graph.json"
    if status.get("graph_sha256") is not None and (
        not graph.is_file() or _file_hash(graph) != status["graph_sha256"]
    ):
        problems.append("graph hash mismatch")
    return problems


def _target(value: str) -> Target:
    try:
        base_url, model = value.split("|", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("target must be URL|MODEL") from exc
    base_url = base_url.rstrip("/")
    parts = urlsplit(base_url)
    if (
        parts.scheme != "http"
        or not parts.hostname
        or parts.path != "/v1"
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
        or not model
    ):
        raise argparse.ArgumentTypeError("target must use http://HOST:PORT/v1|MODEL")
    return Target(base_url, model)


def _request_json(url: str, payload: dict, *, timeout: float) -> dict:
    request = urllib.request.Request(  # noqa: S310 - validated HTTP Ollama target
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    # The URL is built from a validated HTTP Ollama target.
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.load(response)


def _check_targets(targets: list[Target]) -> None:
    from graphify.llm import _validate_ollama_base_url

    for target in targets:
        _validate_ollama_base_url(target.base_url, warn=False)
        data = _request_json(f"{target.api_url}/api/show", {"model": target.model}, timeout=10)
        if not data.get("details"):
            raise RuntimeError(f"Ollama does not recognize {target.model} at {target.base_url}")
        loaded = _request_json(
            f"{target.api_url}/api/generate",
            {"model": target.model, "keep_alive": -1},
            timeout=90,
        )
        if loaded.get("error"):
            raise RuntimeError(f"Ollama could not load {target.model}: {loaded['error']}")
        # A model that silently spills to the CPU defeats the two-GPU setup.
        with urllib.request.urlopen(f"{target.api_url}/api/ps", timeout=10) as response:  # noqa: S310
            running = json.load(response).get("models", [])
        match = next((m for m in running if m.get("model") == target.model), None)
        if not match or not match.get("size") or match.get("size_vram", 0) < match["size"]:
            raise RuntimeError(
                f"{target.model} is not fully loaded on the GPU at {target.base_url}; "
                "free VRAM or reduce its model context before starting Graphify"
            )


def make_handler(pool: TargetPool, timeout: float) -> type[BaseHTTPRequestHandler]:
    """Create the loopback-only OpenAI compatibility handler."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:  # noqa: N802 - http.server interface
            if self.path.rstrip("/") != "/v1/models":
                self._send(404, b"not found", "text/plain")
                return
            body = json.dumps(
                {"object": "list", "data": [{"id": "dual-gpu", "object": "model"}]}
            ).encode()
            self._send(200, body, "application/json")

        def do_POST(self) -> None:  # noqa: N802 - http.server interface
            if self.path != "/v1/chat/completions":
                self._send(404, b"not found", "text/plain")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 16 * 1024 * 1024:
                    self._send(413, b"invalid request size", "text/plain")
                    return
                payload = json.loads(self.rfile.read(length))
                if not isinstance(payload, dict) or payload.get("stream") is True:
                    self._send(400, b"expected a non-streaming JSON request", "text/plain")
                    return
            except (ValueError, json.JSONDecodeError):
                self._send(400, b"invalid JSON", "text/plain")
                return

            index = pool.available.get()
            target = pool.targets[index]
            request_id = uuid.uuid4().hex
            input_hash = hashlib.sha256(
                json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            try:
                pool.journal(
                    {
                        "phase": "started",
                        "request_id": request_id,
                        "gpu": index,
                        "target": target.base_url,
                        "model": target.model,
                        "input_sha256": input_hash,
                    }
                )
            except OSError:
                pool.record(index, failed=True)
                pool.available.put(index)
                self._send(503, b'{"error":"request journal unavailable"}', "application/json")
                return
            payload["model"] = target.model
            request = urllib.request.Request(  # noqa: S310 - validated HTTP Ollama target
                f"{target.base_url}/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json", "X-Graphify-Request-ID": request_id},
            )
            started = time.monotonic()
            failed = False
            status = 502
            body = b'{"error":"upstream response unavailable"}'
            content_type = "application/json"
            try:
                try:
                    # Pool targets were checked before proxy startup.
                    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                        status = response.status
                        body = response.read()
                        content_type = response.headers.get("Content-Type", "application/json")
                except urllib.error.HTTPError as exc:
                    status = exc.code
                    body = exc.read()
                    content_type = exc.headers.get("Content-Type", "application/json")
                    failed = True
            except (OSError, TimeoutError) as exc:
                failed = True
                status = 502
                body = json.dumps({"error": str(exc)}).encode()
                content_type = "application/json"
            finally:
                try:
                    pool.journal(
                        {
                            "phase": "completed",
                            "request_id": request_id,
                            "gpu": index,
                            "http_status": status,
                            "response_sha256": hashlib.sha256(body).hexdigest(),
                            "elapsed_ms": round((time.monotonic() - started) * 1000),
                        }
                    )
                except OSError:
                    failed = True
                    status = 503
                    body = b'{"error":"request journal unavailable"}'
                    content_type = "application/json"
                pool.record(index, failed=failed)
                pool.available.put(index)
                print(
                    f"[dual-gpu] GPU {index}: {time.monotonic() - started:.1f}s"
                    f"{' ERROR' if failed else ''}",
                    file=sys.stderr,
                    flush=True,
                )
            self._send(status, body, content_type, request_id=request_id)

        def _send(
            self,
            status: int,
            body: bytes,
            content_type: str,
            *,
            request_id: str | None = None,
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            if request_id is not None:
                self.send_header("X-Graphify-Run-ID", pool.run_id)
                self.send_header("X-Graphify-Request-ID", request_id)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format: str, *args: object) -> None:
            pass

    return Handler


def run(argv: list[str] | None = None) -> int:
    """Proxy Graphify across two GPUs and report extraction completeness."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="repository to scan")
    parser.add_argument("--out", required=True, type=Path, help="separate output directory")
    parser.add_argument(
        "--target", action="append", type=_target, help="URL|MODEL; pass twice"
    )
    parser.add_argument("--token-budget", type=int, default=3000)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--api-timeout", type=float, default=600)
    parser.add_argument("--no-cluster", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--code-only", action="store_true", help="extract code AST only without GPU/LLM"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="regenerate report and clustering from existing graph",
    )
    args = parser.parse_args(argv)
    source = args.path.resolve()
    output = args.out.resolve()
    if output == source or source in output.parents:
        parser.error("--out must be outside the scanned repository")
    if args.report_only:
        graph_path = output / "graphify-out" / "graph.json"
        if not graph_path.is_file():
            parser.error(f"no existing graph.json found at {graph_path}")
        command = [
            sys.executable,
            "-m",
            "graphify",
            "cluster-only",
            str(output),
            "--graph",
            str(graph_path),
        ]
        return subprocess.run(command).returncode
    if not source.is_dir():
        parser.error(f"repository does not exist: {source}")
    if args.code_only:
        command = [
            sys.executable,
            "-m",
            "graphify",
            "extract",
            str(source),
            "--code-only",
            "--out",
            str(output),
            "--timing",
        ]
        if args.no_cluster:
            command.append("--no-cluster")
        if args.force:
            command.append("--force")
        for pattern in _SEMANTIC_EXCLUDES:
            command.extend(("--exclude", pattern))
        return subprocess.run(command).returncode
    if not args.target or len(args.target) != 2:
        parser.error("exactly two --target arguments are required")
    if args.target[0].base_url == args.target[1].base_url:
        parser.error("the two Ollama targets must be different")
    if args.token_budget <= 0 or args.max_output_tokens <= 0 or args.api_timeout <= 0:
        parser.error("budgets and timeout must be positive")
    _check_semantic_scope(source)
    run_id = uuid.uuid4().hex
    source_sha256, source_files = _source_snapshot(source)
    _check_targets(args.target)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / "dual-gpu-status.json"
    status_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "state": "running",
                "source_sha256": source_sha256,
                "source_files": source_files,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    pool = TargetPool(
        args.target,
        run_id=run_id,
        journal_path=output / "dual-gpu-requests.jsonl",
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(pool, args.api_timeout + 30))
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    command = [
        sys.executable,
        "-m",
        "graphify",
        "extract",
        str(source),
        "--backend",
        "ollama",
        "--model",
        "dual-gpu",
        "--out",
        str(output),
        "--token-budget",
        str(args.token_budget),
        "--max-concurrency",
        "2",
        "--api-timeout",
        str(args.api_timeout),
        "--timing",
    ]
    if args.no_cluster:
        command.append("--no-cluster")
    if args.force:
        command.append("--force")
    for pattern in _SEMANTIC_EXCLUDES:
        command.extend(("--exclude", pattern))
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUNBUFFERED": "1",
            "OLLAMA_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1",
            "OLLAMA_MODEL": "dual-gpu",
            "OLLAMA_API_KEY": "ollama",
            "GRAPHIFY_OLLAMA_PARALLEL": "1",
            "GRAPHIFY_OLLAMA_REASONING_EFFORT": "none",
            "GRAPHIFY_MAX_OUTPUT_TOKENS": str(args.max_output_tokens),
            "GRAPHIFY_OLLAMA_NUM_CTX": "16384",
        }
    )
    print(f"[dual-gpu] Graphify scans {source}; output: {output}", flush=True)
    log_path = output / "dual-gpu-run.log"
    completed_chunks: set[int] = set()
    completion_order: list[int] = []
    expected_chunks = 0
    incomplete = False
    process: subprocess.Popen[str] | None = None
    status = 1
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(  # noqa: S603 - fixed module and argv, no shell
                command,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            if process.stdout is None:
                raise RuntimeError("Graphify output pipe was not created")
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
                match = re.search(r"chunk (\d+)/(\d+) done", line)
                if match:
                    chunk_number = int(match.group(1))
                    completed_chunks.add(chunk_number)
                    completion_order.append(chunk_number)
                    expected_chunks = int(match.group(2))
                if any(
                    marker in line
                    for marker in (
                        "semantic extraction is incomplete",
                        "semantic chunk(s) failed",
                        "produced no nodes and are absent",
                        "chunk(s) failed",
                    )
                ) or ("chunk " in line and " failed:" in line):
                    incomplete = True
            status = process.wait()
    except KeyboardInterrupt:
        status = 130
        print("[dual-gpu] interrupted; stopping Graphify", file=sys.stderr)
    finally:
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    print(
        f"[dual-gpu] requests: GPU 0={pool.counts[0]}, GPU 1={pool.counts[1]}; failures={pool.failures}"
    )
    graph_path = output / "graphify-out" / "graph.json"
    try:
        final_source_sha256, final_source_files = _source_snapshot(source)
    except OSError as exc:
        final_source_sha256, final_source_files = None, None
        print(f"[dual-gpu] source snapshot failed: {exc}", file=sys.stderr)
    if status == 0 and (
        incomplete
        or (expected_chunks and completed_chunks != set(range(1, expected_chunks + 1)))
        or not graph_path.is_file()
        or any(pool.failures)
        or pool.journal_errors
        or final_source_sha256 != source_sha256
    ):
        status = 3
        print(
            "[dual-gpu] incomplete extraction: inspect dual-gpu-run.log; graph is not certified complete",
            file=sys.stderr,
        )
    result = {
        "run_id": run_id,
        "state": "completed" if status == 0 else "incomplete",
        "exit_code": status,
        "graph": str(graph_path) if graph_path.is_file() else None,
        "graph_sha256": _file_hash(graph_path) if graph_path.is_file() else None,
        "journal_sha256": _file_hash(pool.journal_path) if pool.journal_path else None,
        "source_sha256": source_sha256,
        "source_files": source_files,
        "source_sha256_after": final_source_sha256,
        "source_files_after": final_source_files,
        "expected_chunks": expected_chunks,
        "completed_chunks": len(completed_chunks),
        "completion_order": completion_order,
        "requests_per_gpu": pool.counts,
        "failed_requests_per_gpu": pool.failures,
        "journal_errors": pool.journal_errors,
    }
    status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    integrity_errors = verify_run(output)
    if integrity_errors:
        status = 3 if status == 0 else status
        result.update(
            {"state": "incomplete", "exit_code": status, "integrity_errors": integrity_errors}
        )
        status_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"[dual-gpu] integrity check failed: {', '.join(integrity_errors)}", file=sys.stderr)
    if status == 0:
        for target in args.target:
            try:
                _request_json(
                    f"{target.api_url}/api/generate",
                    {"model": target.model, "keep_alive": -1},
                    timeout=30,
                )
            except (OSError, ValueError) as exc:
                print(
                    f"[dual-gpu] model could not stay loaded at {target.base_url}: {exc}",
                    file=sys.stderr,
                )
        if not args.no_cluster and graph_path.is_file():
            report_command = [
                sys.executable,
                "-m",
                "graphify",
                "cluster-only",
                str(output),
                "--graph",
                str(graph_path),
            ]
            print(f"[dual-gpu] generating cluster report for {output}", flush=True)
            report_proc = subprocess.run(report_command, check=False)
            if report_proc.returncode != 0:
                print("[dual-gpu] warning: cluster-only report generation failed", file=sys.stderr)
    return status


if __name__ == "__main__":
    raise SystemExit(run())
