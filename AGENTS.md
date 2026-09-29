# AGENTS.md – Arbeitsregeln & Entwicklungsleitfaden für graphify

Wissensgraph- und Code-Intelligence-Plattform für Agenten und Entwicklungsumgebungen.

---

## 1. Wichtigste Entwickler-Befehle

```bash
# Tests ausführen
uv run --frozen pytest tests/ -q

# Linter & Formatierung
uv run --frozen ruff check .
uv run --frozen ruff format --check .

# Prüflauf via auditcore-runner
auditcore-runner lokal pr

# Dual-GPU-Lauf auf janpow-ai ausführen
graphify-dual-gpu auditcore
graphify-dual-gpu --all

# Internen Wissensgraphen nach Codeänderungen aktualisieren (AST-only, keine LLM-Kosten)
python -m graphify update .
```

---

## 2. Tech-Stack

* **Sprachen & Parser:** Python 3.12+, Tree-Sitter (C-Bindings), Python `ast`
* **Graph & Analyse:** NetworkX, Rust-natives Leiden (`graspologic_native`)
* **Inferenz & Dual-GPU:** Lokale Ollama-Instanzen (Qwen 27B) über HTTP-Proxy
* **Build & Paketmanager:** `uv`, `pyproject.toml`

---

## 3. Architektur & Struktur

* **Pipeline:** `detect` -> `extract` (AST) -> `build` -> `cluster` -> `analyze` -> `export`
* **Details:** Siehe [`ARCHITECTURE.md`](ARCHITECTURE.md) und [`BENCHMARKS.md`](BENCHMARKS.md).

---

## 4. Graphify-Regeln für Agenten

* Nach jeder Codeänderung im Repository: `python -m graphify update .` ausführen, um den Graphen aktuell zu halten.
* Wenn `graphify-out/wiki/index.md` existiert, vorrangig dort navigieren statt Rohdateien zu durchforsten.
* Bei Fragen zur Codebasis: gezielte Graphenabfragen nutzen; den Graphen nicht als Beweis verwenden, wenn es um die Korrektheit des Graphen selbst geht.
