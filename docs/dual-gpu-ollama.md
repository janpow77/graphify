# Zwei lokale Ollama-GPUs für einen Graphify-Lauf

`graphify.dual_gpu` verteilt parallele semantische Chunks auf zwei Ollama-
Instanzen. Graphify erstellt den AST-Anteil einmal, sammelt Chunk-Ergebnisse
nach ihrer ursprünglichen Nummer und baut daraus **einen** Graphen. Ein
schnellerer Endpunkt übernimmt nach jeder Antwort den nächsten Chunk.
Der Standardlauf verarbeitet Programmcode sowie Markdown- und TXT-Dateien.
PDFs, Bilder, Office-Dateien, HTML und YAML werden aus der semantischen
Analyse ausgeschlossen. Vor dem Start prüft der Runner, ob andere
semantische Dateitypen übrig bleiben.

Auf `janpow-ai` gibt es dafür den Befehl:

```bash
graphify-dual-gpu auditcore
graphify-dual-gpu --all
```

`--all` verarbeitet die acht Repositories unter `~/Projekte`, beginnend mit
Auditcore. Die Ergebnisse liegen getrennt von den Repositories unter
`~/graphify-dual-out/<repo>/graphify-out/graph.json`. Standardmäßig erstellt
Graphify beim initialen Lauf außerdem die Cluster und den Bericht
`GRAPH_REPORT.md`.

Weitere Optionen:
* `--no-cluster`: überspringt Cluster- und Berichterstellung für einen reinen Graphen.
* `--code-only`: aktualisiert ausschließlich den Code-AST in Sekunden (ohne GPU/LLM) und behält die semantische Schicht bei.
* `--report-only`: generiert Cluster und `GRAPH_REPORT.md` nachträglich aus einem vorhandenen `graph.json` neu.
* `--force`: erzwingt einen vollständigen Neuaufbau unter Umgehung des inkrementellen Caches.

Das Standard-Token-Budget pro Chunk beträgt 3.000 Tokens (anpassbar über
`--token-budget` oder `GRAPHIFY_TOKEN_BUDGET`). Ein späterer Aufruf mit
demselben Ausgabeverzeichnis nutzt Graphifys inkrementellen Cache.

GPU 0 verwendet `qwen38-27b-gpu0-16k:latest` mit 16.384 Kontext-Tokens.
GPU 1 verwendet `smtek/Qwen3.8-27B:Q3_K_M` mit 24.576 Kontext-Tokens. Beide
Varianten teilen sich dieselben Modellgewichte. Der Host-Befehl legt die
Endpunkte und Modellnamen fest; das Python-Modul kann mit zwei `--target`
Argumenten auch andere lokale Ollama-Instanzen verwenden.

## Prüfen

Jeder Lauf schreibt `dual-gpu-run.log` und `dual-gpu-status.json` in sein
Ausgabeverzeichnis. Die Statusdatei zählt fertige Chunks und Anfragen pro
GPU; `completion_order` zeigt die tatsächliche Rückkehrreihenfolge. Das
Anfragenprotokoll `dual-gpu-requests.jsonl` und der Graph werden per SHA-256
geprüft. Bei fehlgeschlagenen oder ausgelassenen Chunks gibt der Befehl einen
Fehlerstatus zurück, auch wenn Graphify einen Teilgraphen geschrieben hat.
Der Originalgraph im Repository bleibt unangetastet.

Ein einzelner Graphify-Ollama-Lauf arbeitet normalerweise seriell. Der
Verteiler setzt `GRAPHIFY_OLLAMA_PARALLEL=1` und `--max-concurrency 2`; sein
lokaler HTTP-Server bindet ausschließlich an `127.0.0.1` und leitet
nicht-streamende Chat-Anfragen an die beiden angegebenen Ollama-Instanzen
weiter. Quelltext und Dokumente bleiben auf `janpow-ai`.
Der Host-Befehl sperrt einen zweiten gleichzeitigen Dual-GPU-Lauf, damit sich
zwei Graphify-Prozesse nicht gegenseitig die GPU-Zeit wegnehmen.
