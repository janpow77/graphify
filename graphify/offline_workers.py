"""Exchange validated semantic file shards between disconnected GPU workers.

The resulting semantic JSON is an intermediate artifact. Build the AST and final
graph once on the coordinator; never merge same-repository graph.json files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path

from graphify.semantic_cleanup import load_validated_semantic_fragment

_SHA = re.compile(r"^[0-9a-f]{64}$")
_RUN = re.compile(r"^[0-9a-f]{32}$")
_MAX_MANIFEST = 4 * 1024 * 1024


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _atomic_json(path: Path, value: dict, *, overwrite: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".graphify-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _relative(value: str) -> Path:
    path = Path(value)
    if not value or path.is_absolute() or any(p in ("", ".", "..") for p in path.parts):
        raise ValueError(f"unsafe relative path: {value!r}")
    if "\\" in value or path.as_posix() != value:
        raise ValueError(f"non-portable relative path: {value!r}")
    return path


def _read_manifest(path: Path) -> dict:
    if path.stat().st_size > _MAX_MANIFEST:
        raise ValueError("manifest exceeds size limit")
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(data, dict)
        or data.get("schema") != 1
        or not _RUN.fullmatch(str(data.get("run_id", "")))
    ):
        raise ValueError("invalid run manifest")
    files = data.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("manifest must contain files")
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("invalid file entry")
        name = entry.get("path")
        if not isinstance(name, str):
            raise ValueError("invalid file path")
        _relative(name)
        if name in seen or not _SHA.fullmatch(str(entry.get("sha256", ""))):
            raise ValueError(f"duplicate path or invalid digest: {name}")
        if type(entry.get("shard")) is not int or entry["shard"] < 0:
            raise ValueError(f"invalid shard for {name}")
        seen.add(name)
    if files != sorted(files, key=lambda item: item["path"]):
        raise ValueError("manifest files must be sorted")
    if type(data.get("shards")) is not int or data["shards"] < 1:
        raise ValueError("invalid shard count")
    if any(entry["shard"] >= data["shards"] for entry in files):
        raise ValueError("shard index out of range")
    models = data.get("models")
    if (
        not isinstance(models, list)
        or len(models) != data["shards"]
        or not all(isinstance(model, str) and model for model in models)
    ):
        raise ValueError("one model name per shard is required")
    if not _SHA.fullmatch(str(data.get("weights_sha256", ""))):
        raise ValueError("model weights digest missing")
    if not _SHA.fullmatch(str(data.get("prompt_sha256", ""))):
        raise ValueError("prompt digest missing")
    token_budget = data.get("token_budget", 1200)
    if type(token_budget) is not int or token_budget < 1:
        raise ValueError("invalid token budget")
    allocation = data.get("allocation")
    if allocation is not None and allocation != "estimated-chunks-greedy-v1":
        raise ValueError("unknown shard allocation strategy")
    for entry in files:
        estimate = entry.get("estimated_chunks")
        if estimate is not None and (type(estimate) is not int or estimate < 1):
            raise ValueError(f"invalid chunk estimate for {entry['path']}")
    return data


def _safe_file(root: Path, name: str, digest: str) -> Path:
    path = root / _relative(name)
    if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"source missing or unsafe: {name}")
    if _digest(path) != digest:
        raise ValueError(f"source changed: {name}")
    return path


def _estimated_chunks(path: Path, token_budget: int) -> int:
    """Estimate worker cost using the extractor's own chunking rules."""
    from graphify.llm import _pack_chunks_by_tokens, expand_oversized_files

    cap = min(20_000, max(1, token_budget * 4 - 160))
    chunks = _pack_chunks_by_tokens(expand_oversized_files([path], cap), token_budget)
    return max(1, len(chunks))


def _assign_by_estimated_chunks(entries: list[dict], shard_count: int) -> list[dict]:
    """Greedily balance largest files first while preserving deterministic ties."""
    loads = [0] * shard_count
    assigned: list[dict | None] = [None] * len(entries)
    order = sorted(
        range(len(entries)),
        key=lambda index: (-entries[index]["estimated_chunks"], entries[index]["path"]),
    )
    for index in order:
        shard = min(range(shard_count), key=lambda candidate: (loads[candidate], candidate))
        entry = {**entries[index], "shard": shard}
        assigned[index] = entry
        loads[shard] += entry["estimated_chunks"]
    return [entry for entry in assigned if entry is not None]


def create(
    root: Path,
    output: Path,
    files: list[str],
    models: list[str],
    weights_sha256: str,
    token_budget: int = 1200,
) -> dict:
    """Create a verified manifest for semantically extracted source files."""
    from graphify.detect import detect
    from graphify.dual_gpu import _SEMANTIC_EXCLUDES
    from graphify.llm import _extraction_system

    if output.exists():
        raise ValueError(f"manifest already exists: {output}")
    shards = len(models)
    if (
        shards < 1
        or not files
        or not _SHA.fullmatch(weights_sha256)
        or type(token_budget) is not int
        or token_budget < 1
    ):
        raise ValueError("provide files, models, and a SHA-256 weight digest")
    root = root.resolve()
    names = sorted(set(files))
    if any(Path(name).suffix.lower() not in {".md", ".txt"} for name in names):
        raise ValueError("only .md and .txt semantic files are supported")
    detection = detect(
        root, cache_root=output.parent.resolve(), extra_excludes=list(_SEMANTIC_EXCLUDES)
    )
    if detection.get("walk_errors"):
        raise ValueError("repository scan was incomplete")
    available = {
        Path(path).resolve().relative_to(root).as_posix()
        for kind in ("document", "paper", "image")
        for path in detection["files"].get(kind, [])
        if Path(path).suffix.lower() in {".md", ".txt"}
    }
    if set(names) != available:
        raise ValueError("plan must contain every detected .md/.txt semantic source")
    entries = []
    for index, name in enumerate(names):
        relative = _relative(name)
        path = root / relative
        if not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(root):
            raise ValueError(f"source missing or unsafe: {name}")
        entries.append(
            {
                "path": name,
                "sha256": _digest(path),
                "estimated_chunks": _estimated_chunks(path, token_budget),
            }
        )
    entries = _assign_by_estimated_chunks(entries, shards)
    prompt = _extraction_system()
    manifest = {
        "schema": 1,
        "run_id": uuid.uuid4().hex,
        "models": models,
        "weights_sha256": weights_sha256,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "shards": shards,
        "token_budget": token_budget,
        "allocation": "estimated-chunks-greedy-v1",
        "files": entries,
    }
    _atomic_json(output, manifest, overwrite=False)
    return manifest


def _artifact_paths(directory: Path, run_id: str, name: str) -> tuple[Path, Path]:
    file_id = hashlib.sha256(name.encode()).hexdigest()
    base = directory / run_id
    return base / f"{file_id}.json", base / f"{file_id}.receipt.json"


def _check_receipt(manifest: dict, entry: dict, directory: Path) -> dict:
    fragment_path, receipt_path = _artifact_paths(directory, manifest["run_id"], entry["path"])
    if (
        not fragment_path.is_file()
        or not receipt_path.is_file()
        or fragment_path.is_symlink()
        or receipt_path.is_symlink()
    ):
        raise ValueError(f"missing result: {entry['path']}")
    if receipt_path.stat().st_size > 8192:
        raise ValueError(f"oversized receipt: {entry['path']}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    expected = {
        "run_id": manifest["run_id"],
        "source_path": entry["path"],
        "source_sha256": entry["sha256"],
        "shard": entry["shard"],
        "model": manifest["models"][entry["shard"]],
        "weights_sha256": manifest["weights_sha256"],
        "prompt_sha256": manifest["prompt_sha256"],
        "fragment_sha256": _digest(fragment_path),
    }
    for key in ("token_budget", "allocation"):
        if key in manifest:
            expected[key] = manifest[key]
    if "estimated_chunks" in entry:
        expected["estimated_chunks"] = entry["estimated_chunks"]
    if receipt != expected:
        raise ValueError(f"result identity/hash mismatch: {entry['path']}")
    fragment, errors = load_validated_semantic_fragment(fragment_path)
    if errors or fragment is None:
        raise ValueError(f"invalid fragment for {entry['path']}: {errors[:2]}")
    for bucket in ("nodes", "edges", "hyperedges"):
        for item in fragment.get(bucket, []):
            if item.get("source_file") != entry["path"]:
                raise ValueError(f"wrong source_file in {entry['path']}")
    if not fragment.get("nodes") and not fragment.get("hyperedges"):
        raise ValueError(f"empty semantic result: {entry['path']}")
    return fragment


def _check_weights(endpoint: str, model: str, expected: str) -> None:
    """Compare the Ollama GGUF/blob identity, not the tag or Modelfile hash."""
    request = urllib.request.Request(  # noqa: S310 - validated Ollama endpoint
        f"{endpoint.removesuffix('/v1')}/api/show",
        data=json.dumps({"model": model}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
        shown = json.load(response)
    modelfile = shown.get("modelfile", "")
    match = re.search(r"^FROM\s+\S*/sha256-([0-9a-f]{64})\s*$", modelfile, re.MULTILINE)
    if match is None or match.group(1) != expected:
        raise ValueError(f"Ollama model weights differ or cannot be verified: {model}")


def run_shard(
    manifest_path: Path,
    root: Path,
    directory: Path,
    shard: int,
    endpoint: str,
    token_budget: int | None = None,
) -> int:
    """Extract one shard while checking model and source identities."""
    from graphify.dual_gpu import Target, _check_targets
    from graphify.llm import (
        BACKENDS,
        _extraction_system,
        _partial_source_files,
        extract_corpus_parallel,
    )
    from graphify.cache import scope_semantic_result

    manifest = _read_manifest(manifest_path)
    token_budget = token_budget or manifest.get("token_budget", 1200)
    if not 0 <= shard < manifest["shards"] or token_budget < 1:
        raise ValueError("invalid shard or token budget")
    if hashlib.sha256(_extraction_system().encode()).hexdigest() != manifest["prompt_sha256"]:
        raise ValueError("worker extraction prompt differs from manifest")
    # Refuse inference if Ollama did not load the entire model in GPU VRAM.
    model = manifest["models"][shard]
    target = Target(endpoint.rstrip("/"), model)
    _check_targets([target])
    _check_weights(endpoint, model, manifest["weights_sha256"])
    os.environ["OLLAMA_BASE_URL"] = endpoint.rstrip("/")
    os.environ["OLLAMA_MODEL"] = model
    os.environ["OLLAMA_API_KEY"] = "ollama"
    os.environ["GRAPHIFY_OLLAMA_REASONING_EFFORT"] = "none"
    os.environ["GRAPHIFY_OLLAMA_NUM_CTX"] = "16384"
    os.environ["GRAPHIFY_MAX_OUTPUT_TOKENS"] = "4096"
    # BACKENDS is initialized when graphify.llm is imported above. The worker
    # selects its endpoint afterwards, so refresh the cached provider settings.
    BACKENDS["ollama"]["base_url"] = endpoint.rstrip("/")
    BACKENDS["ollama"]["default_model"] = model
    root = root.resolve()
    complete = 0
    progress_path = directory / manifest["run_id"] / f"shard-{shard}.progress.json"
    failures_path = directory / manifest["run_id"] / f"shard-{shard}.failures.json"
    failures: list[str] = []
    _atomic_json(failures_path, {"run_id": manifest["run_id"], "shard": shard, "paths": failures})
    for entry in manifest["files"]:
        if entry["shard"] != shard:
            continue
        path = _safe_file(root, entry["path"], entry["sha256"])
        try:
            _check_receipt(manifest, entry, directory)
        except ValueError:
            fragment_path, receipt_path = _artifact_paths(
                directory, manifest["run_id"], entry["path"]
            )
            # A mismatched existing result is an integrity failure, not a cache miss.
            if fragment_path.exists() or receipt_path.exists():
                raise
        else:
            complete += 1
            continue

        attempt = 0

        def on_chunk_done(index: int, total: int, result: dict) -> None:
            _atomic_json(
                progress_path,
                {
                    "run_id": manifest["run_id"],
                    "shard": shard,
                    "current_file": entry["path"],
                    "chunks_done": index + 1,
                    "chunks_total": total,
                    "failed_chunks": result.get("failed_chunks", 0),
                    "attempt": attempt,
                },
            )

        from graphify.semantic_cleanup import validate_semantic_fragment

        for attempt in range(3):
            _atomic_json(
                progress_path,
                {
                    "run_id": manifest["run_id"],
                    "shard": shard,
                    "current_file": entry["path"],
                    "chunks_done": 0,
                    "attempt": attempt,
                },
            )
            cache_root = directory / manifest["run_id"]
            if attempt:
                cache_root = (
                    cache_root
                    / "retry"
                    / hashlib.sha256(entry["path"].encode()).hexdigest()
                    / str(attempt)
                )
            result = extract_corpus_parallel(
                [path],
                backend="ollama",
                model=model,
                root=root,
                token_budget=token_budget,
                max_concurrency=1,
                cache_root=cache_root,
                on_chunk_done=on_chunk_done,
            )
            _safe_file(root, entry["path"], entry["sha256"])
            if result.get("failed_chunks") or _partial_source_files(result):
                print(
                    f"[graphify] retrying incomplete extraction: {entry['path']}", file=sys.stderr
                )
                continue
            scope_semantic_result(result, root=root, allowed_source_files=[path])
            fragment = {key: result.get(key, []) for key in ("nodes", "edges", "hyperedges")}
            fragment.update({key: result.get(key, 0) for key in ("input_tokens", "output_tokens")})
            for bucket in ("nodes", "edges", "hyperedges"):
                for item in fragment[bucket]:
                    source = item.get("source_file")
                    if source not in (None, "", entry["path"], str(path)):
                        raise ValueError(f"unexpected source_file in {entry['path']}")
                    item["source_file"] = entry["path"]
            if not validate_semantic_fragment(fragment) and (
                fragment["nodes"] or fragment["hyperedges"]
            ):
                break
            print(f"[graphify] retrying invalid or empty result: {entry['path']}", file=sys.stderr)
        else:
            failures.append(entry["path"])
            _atomic_json(
                failures_path,
                {"run_id": manifest["run_id"], "shard": shard, "paths": failures},
            )
            print(
                f"[graphify] postponed after 3 invalid attempts: {entry['path']}", file=sys.stderr
            )
            continue
        fragment_path, receipt_path = _artifact_paths(directory, manifest["run_id"], entry["path"])
        _check_targets([target])
        _atomic_json(fragment_path, fragment)
        receipt = {
            "run_id": manifest["run_id"],
            "source_path": entry["path"],
            "source_sha256": entry["sha256"],
            "shard": shard,
            "model": model,
            "weights_sha256": manifest["weights_sha256"],
            "prompt_sha256": manifest["prompt_sha256"],
            "fragment_sha256": _digest(fragment_path),
        }
        for key in ("token_budget", "allocation"):
            if key in manifest:
                receipt[key] = manifest[key]
        if "estimated_chunks" in entry:
            receipt["estimated_chunks"] = entry["estimated_chunks"]
        _atomic_json(receipt_path, receipt)
        progress_path.unlink(missing_ok=True)
        complete += 1
    if failures:
        raise RuntimeError(f"{len(failures)} semantic file(s) still incomplete: {failures[:3]}")
    failures_path.unlink(missing_ok=True)
    return complete


def merge(manifest_path: Path, root: Path, directory: Path, output: Path) -> dict:
    """Combine every verified shard into one semantic intermediate file."""
    manifest = _read_manifest(manifest_path)
    if output.exists():
        raise ValueError(f"output already exists: {output}")
    root = root.resolve()
    fragments = []
    for entry in manifest["files"]:
        _safe_file(root, entry["path"], entry["sha256"])
        fragments.append(_check_receipt(manifest, entry, directory))
    merged: dict = {
        "nodes": [],
        "edges": [],
        "hyperedges": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }
    node_ids: dict[str, dict] = {}
    for fragment in fragments:
        for node in fragment.get("nodes", []):
            prior = node_ids.get(node["id"])
            if prior is not None:
                if prior != node:
                    raise ValueError(f"conflicting node ID: {node['id']}")
                continue
            node_ids[node["id"]] = node
            merged["nodes"].append(node)
        for bucket in ("edges", "hyperedges"):
            merged[bucket].extend(fragment.get(bucket, []))
        for key in ("input_tokens", "output_tokens"):
            value = fragment.get(key, 0)
            if type(value) not in (int, float) or value < 0:
                raise ValueError(f"invalid {key}")
            merged[key] += value
    _atomic_json(output, merged)
    return merged


def import_results(manifest_path: Path, incoming: Path, collected: Path) -> int:
    """Validate a staged transfer, then copy complete pairs without overwriting."""
    manifest = _read_manifest(manifest_path)
    accepted = 0
    for entry in manifest["files"]:
        source_fragment, source_receipt = _artifact_paths(
            incoming, manifest["run_id"], entry["path"]
        )
        if not source_fragment.exists() and not source_receipt.exists():
            continue
        _check_receipt(manifest, entry, incoming)
        target_fragment, target_receipt = _artifact_paths(
            collected, manifest["run_id"], entry["path"]
        )
        if target_fragment.exists() or target_receipt.exists():
            if target_fragment.exists() and _digest(source_fragment) != _digest(target_fragment):
                raise ValueError(f"conflicting result for {entry['path']}")
            if target_receipt.exists() and _digest(source_receipt) != _digest(target_receipt):
                raise ValueError(f"conflicting receipt for {entry['path']}")
            if target_fragment.exists() and target_receipt.exists():
                _check_receipt(manifest, entry, collected)
                continue
        target_fragment.parent.mkdir(parents=True, exist_ok=True)
        for source, target in (
            (source_fragment, target_fragment),
            (source_receipt, target_receipt),
        ):
            if target.exists():
                continue
            descriptor, temporary = tempfile.mkstemp(prefix=".graphify-import-", dir=target.parent)
            try:
                with os.fdopen(descriptor, "wb") as output, source.open("rb") as input_stream:
                    shutil.copyfileobj(input_stream, output)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        accepted += 1
        _check_receipt(manifest, entry, collected)
    return accepted


def finalize(manifest_path: Path, root: Path, collected: Path, output: Path) -> int:
    """Seed Graphify's semantic cache and build AST/final graph without LLM calls."""
    from graphify.cache import check_semantic_cache, save_semantic_cache
    from graphify.detect import detect
    from graphify.dual_gpu import _SEMANTIC_EXCLUDES
    from graphify.llm import _extraction_system

    manifest = _read_manifest(manifest_path)
    root = root.resolve()
    output = output.resolve()
    if output == root or root in output.parents:
        raise ValueError("output must be outside the repository")
    prompt = _extraction_system()
    if hashlib.sha256(prompt.encode()).hexdigest() != manifest["prompt_sha256"]:
        raise ValueError("finalizer prompt differs from manifest")
    detection = detect(root, cache_root=output, extra_excludes=list(_SEMANTIC_EXCLUDES))
    if detection.get("walk_errors"):
        raise ValueError("repository scan was incomplete")
    files_by_type = detection.get("files", {})
    all_semantic = {
        Path(path).resolve().relative_to(root).as_posix()
        for kind in ("document", "paper", "image")
        for path in files_by_type.get(kind, [])
    }
    excluded = sorted(
        "/" + name for name in all_semantic if Path(name).suffix.lower() not in {".md", ".txt"}
    )
    scoped = detect(root, cache_root=output, extra_excludes=[*_SEMANTIC_EXCLUDES, *excluded])
    if scoped.get("walk_errors"):
        raise ValueError("scoped repository scan was incomplete")
    detected = {
        Path(path).resolve().relative_to(root).as_posix()
        for kind in ("document", "paper", "image")
        for path in scoped["files"].get(kind, [])
    }
    if set(scoped["files"].get("code", [])) != set(files_by_type.get("code", [])):
        raise ValueError("semantic exclusions removed code sources")
    expected = {entry["path"] for entry in manifest["files"]}
    if detected != expected:
        raise ValueError(
            f"semantic corpus differs from manifest: {len(expected - detected)} missing, "
            f"{len(detected - expected)} additional files"
        )
    fragments = []
    for entry in manifest["files"]:
        _safe_file(root, entry["path"], entry["sha256"])
        fragments.append(_check_receipt(manifest, entry, collected))
    for entry, fragment in zip(manifest["files"], fragments):
        count = save_semantic_cache(
            fragment["nodes"],
            fragment["edges"],
            fragment.get("hyperedges", []),
            root=root,
            cache_root=output,
            prompt=prompt,
            allowed_source_files=[root / entry["path"]],
        )
        if count != 1:
            raise ValueError(f"could not cache semantic result: {entry['path']}")
    _, _, _, misses = check_semantic_cache(
        [str(root / entry["path"]) for entry in manifest["files"]],
        root=root,
        cache_root=output,
        prompt=prompt,
    )
    if misses:
        raise ValueError(f"semantic cache misses remain: {misses[:3]}")
    environment = os.environ.copy()
    # If a future code path unexpectedly asks an LLM, it cannot reach a model.
    environment.update({"OLLAMA_BASE_URL": "http://127.0.0.1:1/v1", "OLLAMA_API_KEY": "ollama"})
    command = [
        sys.executable,
        "-m",
        "graphify",
        "extract",
        str(root),
        "--backend",
        "ollama",
        "--model",
        manifest["models"][0],
        "--out",
        str(output),
        "--no-cluster",
    ]
    for pattern in (*_SEMANTIC_EXCLUDES, *excluded):
        command.extend(("--exclude", pattern))
    return subprocess.run(command, env=environment, check=False).returncode  # noqa: S603


def main(argv: list[str] | None = None) -> int:
    """Dispatch the offline worker command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("root", type=Path)
    plan.add_argument("manifest", type=Path)
    plan.add_argument("files", nargs="+")
    plan.add_argument("--model", required=True, action="append")
    plan.add_argument("--weights-sha256", required=True)
    plan.add_argument("--token-budget", type=int, default=1200)
    worker = commands.add_parser("run-shard")
    worker.add_argument("manifest", type=Path)
    worker.add_argument("root", type=Path)
    worker.add_argument("results", type=Path)
    worker.add_argument("--shard", type=int, required=True)
    worker.add_argument("--endpoint", required=True)
    worker.add_argument("--token-budget", type=int)
    join = commands.add_parser("merge")
    join.add_argument("manifest", type=Path)
    join.add_argument("root", type=Path)
    join.add_argument("results", type=Path)
    join.add_argument("output", type=Path)
    ingest = commands.add_parser("import")
    ingest.add_argument("manifest", type=Path)
    ingest.add_argument("incoming", type=Path)
    ingest.add_argument("collected", type=Path)
    finish = commands.add_parser("finalize")
    finish.add_argument("manifest", type=Path)
    finish.add_argument("root", type=Path)
    finish.add_argument("collected", type=Path)
    finish.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            result = create(
                args.root,
                args.manifest,
                args.files,
                args.model,
                args.weights_sha256,
                args.token_budget,
            )
            print(result["run_id"])
        elif args.command == "run-shard":
            print(
                run_shard(
                    args.manifest,
                    args.root,
                    args.results,
                    args.shard,
                    args.endpoint,
                    args.token_budget,
                )
            )
        elif args.command == "import":
            print(import_results(args.manifest, args.incoming, args.collected))
        elif args.command == "finalize":
            return finalize(args.manifest, args.root, args.collected, args.output)
        else:
            result = merge(args.manifest, args.root, args.results, args.output)
            print(f"{len(result['nodes'])} nodes, {len(result['edges'])} edges")
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"offline-workers: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
