"""Integrity and resume boundaries for disconnected semantic workers."""

# ruff: noqa: S101 - assertions are the checks in this test module

import json
import subprocess
import sys

import pytest

from graphify.offline_workers import (
    _artifact_paths,
    _assign_by_estimated_chunks,
    _atomic_json,
    _digest,
    create,
    finalize,
    import_results,
    merge,
    run_shard,
)


def _fixture(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    for name in ("a.md", "b.md"):
        (root / name).write_text(f"# {name}\n", encoding="utf-8")
    manifest_path = tmp_path / "job.json"
    manifest = create(root, manifest_path, ["a.md", "b.md"], ["model-a", "model-b"], "a" * 64)
    incoming = tmp_path / "incoming"
    for entry in manifest["files"]:
        fragment_path, receipt_path = _artifact_paths(incoming, manifest["run_id"], entry["path"])
        fragment = {
            "nodes": [{"id": entry["path"].replace(".", ":"), "source_file": entry["path"]}],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 2,
            "output_tokens": 1,
        }
        _atomic_json(fragment_path, fragment)
        receipt = {
                "run_id": manifest["run_id"],
                "source_path": entry["path"],
                "source_sha256": entry["sha256"],
                "shard": entry["shard"],
                "model": manifest["models"][entry["shard"]],
                "weights_sha256": manifest["weights_sha256"],
                "prompt_sha256": manifest["prompt_sha256"],
                "fragment_sha256": _digest(fragment_path),
            }
        receipt.update(
            {
                key: value
                for key, value in {
                    "token_budget": manifest.get("token_budget"),
                    "allocation": manifest.get("allocation"),
                    "estimated_chunks": entry.get("estimated_chunks"),
                }.items()
                if value is not None
            }
        )
        _atomic_json(receipt_path, receipt)
    return root, manifest_path, manifest, incoming


def test_import_is_idempotent_and_merge_is_complete(tmp_path):
    """Repeated imports preserve one complete semantic result."""
    root, manifest_path, _, incoming = _fixture(tmp_path)
    collected = tmp_path / "collected"
    assert import_results(manifest_path, incoming, collected) == 2
    assert import_results(manifest_path, incoming, collected) == 0
    result = merge(manifest_path, root, collected, tmp_path / "semantic.json")
    assert [n["source_file"] for n in result["nodes"]] == ["a.md", "b.md"]
    assert result["input_tokens"] == 4


def test_plan_ignores_office_files_and_generated_markdown(tmp_path, monkeypatch):
    """Office conversion must not widen a code-and-text manifest."""
    import graphify.detect as detector

    root = tmp_path / "repo"
    root.mkdir()
    (root / "notes.md").write_text("# Real source\n", encoding="utf-8")
    (root / "template.docx").write_bytes(b"office placeholder")
    generated = root / "graphify-out" / "converted" / "template.md"
    generated.parent.mkdir(parents=True)
    generated.write_text("# Generated source\n", encoding="utf-8")
    monkeypatch.setattr(
        detector,
        "convert_office_file",
        lambda *args, **kwargs: pytest.fail("Office conversion must be excluded"),
    )

    manifest = create(root, tmp_path / "job.json", ["notes.md"], ["model"], "a" * 64)

    assert [entry["path"] for entry in manifest["files"]] == ["notes.md"]


def test_changed_source_blocks_merge(tmp_path):
    """A changed source file cannot be merged with its old result."""
    root, manifest_path, _, incoming = _fixture(tmp_path)
    (root / "a.md").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="source changed"):
        merge(manifest_path, root, incoming, tmp_path / "semantic.json")
    assert not (tmp_path / "semantic.json").exists()


def test_tampered_fragment_blocks_import(tmp_path):
    """A modified fragment fails its receipt hash check."""
    _, manifest_path, manifest, incoming = _fixture(tmp_path)
    fragment_path, _ = _artifact_paths(incoming, manifest["run_id"], "a.md")
    fragment_path.write_text(json.dumps({"nodes": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="identity/hash mismatch"):
        import_results(manifest_path, incoming, tmp_path / "collected")


def test_missing_shard_blocks_merge(tmp_path):
    """A missing shard prevents an incomplete merge."""
    root, manifest_path, manifest, incoming = _fixture(tmp_path)
    fragment_path, receipt_path = _artifact_paths(incoming, manifest["run_id"], "b.md")
    fragment_path.unlink()
    receipt_path.unlink()
    with pytest.raises(ValueError, match="missing result"):
        merge(manifest_path, root, incoming, tmp_path / "semantic.json")


def test_import_recovers_interrupted_file_pair(tmp_path):
    """An interrupted import can resume from a matching fragment."""
    _, manifest_path, manifest, incoming = _fixture(tmp_path)
    collected = tmp_path / "collected"
    source_fragment, _ = _artifact_paths(incoming, manifest["run_id"], "a.md")
    target_fragment, _ = _artifact_paths(collected, manifest["run_id"], "a.md")
    target_fragment.parent.mkdir(parents=True)
    target_fragment.write_bytes(source_fragment.read_bytes())
    assert import_results(manifest_path, incoming, collected) == 2
    assert import_results(manifest_path, incoming, collected) == 0


def test_conflicting_duplicate_blocks_import(tmp_path):
    """Two different results for one source file are rejected."""
    _, manifest_path, manifest, incoming = _fixture(tmp_path)
    collected = tmp_path / "collected"
    assert import_results(manifest_path, incoming, collected) == 2
    fragment_path, receipt_path = _artifact_paths(incoming, manifest["run_id"], "a.md")
    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    fragment["nodes"][0]["label"] = "changed"
    _atomic_json(fragment_path, fragment)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt["fragment_sha256"] = _digest(fragment_path)
    _atomic_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="conflicting result"):
        import_results(manifest_path, incoming, collected)


def test_worker_scopes_model_inventions_and_checks_gpu_twice(tmp_path, monkeypatch):
    """A worker drops invented sources and rechecks the GPU after extraction."""
    root, manifest_path, manifest, _ = _fixture(tmp_path)
    import graphify.dual_gpu as dual_gpu
    import graphify.llm as llm
    import graphify.offline_workers as workers

    calls = []
    monkeypatch.setitem(llm.BACKENDS["ollama"], "base_url", "http://127.0.0.1:1/v1")
    monkeypatch.setattr(dual_gpu, "_check_targets", lambda targets: calls.append(targets[0]))
    monkeypatch.setattr(workers, "_check_weights", lambda *args: None)
    monkeypatch.setattr(
        llm,
        "extract_corpus_parallel",
        lambda files, **kwargs: {
            "nodes": [
                {"id": "real", "source_file": "a.md"},
                {"id": "invented", "source_file": "outside.md"},
            ],
            "edges": [],
            "hyperedges": [],
            "input_tokens": 1,
            "output_tokens": 1,
            "failed_chunks": 0,
        },
    )
    assert (
        run_shard(manifest_path, root, tmp_path / "worker", 0, "http://127.0.0.1:11436/v1", 1200)
        == 1
    )
    assert len(calls) == 2
    assert calls[0].model == manifest["models"][0]
    assert llm.BACKENDS["ollama"]["base_url"] == "http://127.0.0.1:11436/v1"
    fragment_path, _ = _artifact_paths(tmp_path / "worker", manifest["run_id"], "a.md")
    fragment = json.loads(fragment_path.read_text(encoding="utf-8"))
    assert [node["id"] for node in fragment["nodes"]] == ["real"]


def test_worker_retries_empty_result_with_fresh_cache(tmp_path, monkeypatch):
    """Ein leeres Modellresultat darf den Shard nicht sofort abbrechen."""
    root, manifest_path, manifest, _ = _fixture(tmp_path)
    import graphify.dual_gpu as dual_gpu
    import graphify.llm as llm
    import graphify.offline_workers as workers

    monkeypatch.setattr(dual_gpu, "_check_targets", lambda targets: None)
    monkeypatch.setattr(workers, "_check_weights", lambda *args: None)
    cache_roots = []

    def extract(files, **kwargs):
        cache_roots.append(kwargs["cache_root"])
        kwargs["on_chunk_done"](0, 1, {"failed_chunks": 0})
        return {
            "nodes": [] if len(cache_roots) == 1 else [{"id": "real", "source_file": "a.md"}],
            "edges": [],
            "hyperedges": [],
            "failed_chunks": 0,
        }

    monkeypatch.setattr(llm, "extract_corpus_parallel", extract)
    directory = tmp_path / "worker"
    assert run_shard(manifest_path, root, directory, 0, "http://127.0.0.1:11436/v1", 1200) == 1
    assert len(cache_roots) == 2 and cache_roots[0] != cache_roots[1]
    assert not (directory / manifest["run_id"] / "shard-0.progress.json").exists()


def test_worker_continues_after_invalid_file_without_certifying_shard(tmp_path, monkeypatch):
    """Ein defektes Dokument blockiert weitere Dateien nicht und bleibt als Fehler sichtbar."""
    root, manifest_path, manifest, _ = _fixture(tmp_path)
    import graphify.dual_gpu as dual_gpu
    import graphify.llm as llm
    import graphify.offline_workers as workers

    manifest["files"][1]["shard"] = 0
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(dual_gpu, "_check_targets", lambda targets: None)
    monkeypatch.setattr(workers, "_check_weights", lambda *args: None)

    def extract(files, **kwargs):
        name = files[0].name
        return {
            "nodes": [] if name == "a.md" else [{"id": "b", "source_file": "b.md"}],
            "edges": [],
            "hyperedges": [],
            "failed_chunks": 0,
        }

    monkeypatch.setattr(llm, "extract_corpus_parallel", extract)
    directory = tmp_path / "worker"
    with pytest.raises(RuntimeError, match="still incomplete"):
        run_shard(manifest_path, root, directory, 0, "http://127.0.0.1:11436/v1", 1200)
    assert _artifact_paths(directory, manifest["run_id"], "b.md")[1].is_file()
    failures = json.loads((directory / manifest["run_id"] / "shard-0.failures.json").read_text())
    assert failures["paths"] == ["a.md"]


def test_finalize_uses_cache_for_offline_graph_build(tmp_path):
    """Final graph construction consumes the verified semantic cache."""
    root, manifest_path, _, incoming = _fixture(tmp_path)
    (root / "ignored.html").write_text("<h1>Do not send to LLM</h1>", encoding="utf-8")
    (root / "ignored.yaml").write_text("title: also ignored\n", encoding="utf-8")
    (root / "Macro.bas").write_text("Sub Run()\nEnd Sub\n", encoding="utf-8")
    collected = tmp_path / "collected"
    assert import_results(manifest_path, incoming, collected) == 2
    subprocess.run(
        [
            sys.executable,
            "-m",
            "graphify",
            "extract",
            str(root),
            "--code-only",
            "--out",
            str(tmp_path / "out"),
            "--no-cluster",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert finalize(manifest_path, root, collected, tmp_path / "out") == 0
    graph = tmp_path / "out" / "graphify-out" / "graph.json"
    assert graph.is_file()
    sources = {node.get("source_file", "") for node in json.loads(graph.read_text())["nodes"]}
    assert any("Macro.bas" in source for source in sources)
    assert not any("ignored.html" in source or "ignored.yaml" in source for source in sources)


def test_plan_rejects_non_text_semantic_sources(tmp_path):
    """The optional GPU plan admits only Markdown and plain text."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "index.md").write_text("# Index\n", encoding="utf-8")
    (root / "image.png").write_bytes(b"not an image")
    with pytest.raises(ValueError, match="only .md and .txt"):
        create(root, tmp_path / "job.json", ["index.md", "image.png"], ["model"], "a" * 64)


def test_single_shard_plan(tmp_path):
    """A future single-GPU worker can own the complete text corpus."""
    root = tmp_path / "repo"
    root.mkdir()
    (root / "index.md").write_text("# Index\n", encoding="utf-8")
    (root / "notes.txt").write_text("Notes\n", encoding="utf-8")
    manifest = create(
        root, tmp_path / "job.json", ["index.md", "notes.txt"], ["local-model"], "a" * 64
    )
    assert manifest["shards"] == 1
    assert {entry["shard"] for entry in manifest["files"]} == {0}


def test_chunk_assignment_balances_large_files_first_deterministically():
    """Estimated chunk work, not file count, determines the shard assignment."""
    entries = [
        {"path": "a.md", "estimated_chunks": 12},
        {"path": "b.md", "estimated_chunks": 8},
        {"path": "c.md", "estimated_chunks": 4},
        {"path": "d.md", "estimated_chunks": 2},
    ]

    assigned = _assign_by_estimated_chunks(entries, 2)

    loads = [
        sum(entry["estimated_chunks"] for entry in assigned if entry["shard"] == shard)
        for shard in range(2)
    ]
    assert loads == [14, 12]
    assert [entry["path"] for entry in assigned] == ["a.md", "b.md", "c.md", "d.md"]
    assert _assign_by_estimated_chunks(entries, 2) == assigned
