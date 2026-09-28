# Offline-Arbeit auf GPU-Rechnern

`graphify.offline_workers` verteilt **`.md`- und `.txt`-Dateien** eines
Repository-Snapshots. Jeder Worker liest dieselben relativen Pfade und
Dateiinhalte, arbeitet auf seinem lokalen Ollama-GPU-Endpunkt weiter, wenn die
Verbindung zum Koordinator abbricht, und schreibt pro Datei ein Ergebnis mit
Receipt. Der Worker baut keinen kompletten Graphen. AST und finaler Graph
werden einmal auf dem Koordinator erstellt. `merge-graphs` ist dafür nicht
geeignet: Es behandelt Eingaben als verschiedene Repositories.

## Ablauf

Die Dateien im Manifest müssen relativ zur Repo-Wurzel angegeben sein. `plan`
prüft, dass **alle** von Graphify erkannten `.md`/`.txt`-Quellen enthalten
sind. Code, einschließlich VBA-`.bas`, läuft auf dem Koordinator über den
normalen AST-Pfad. PDF, Bilder, Office-Dateien, HTML und YAML gelangen nicht
in diesen semantischen GPU-Lauf. Beispiel mit zwei Textdateien:

```bash
graphify extract /data/repo --code-only --out /data/graphify-result --no-cluster
```

Dieser Code-Lauf verwendet keine GPU und keinen LLM. Für die optionale
semantische Ergänzung von `.md`/`.txt` folgt anschließend:

```bash
python -m graphify.offline_workers plan /data/repo /data/jobs/job.json \
  README.md docs/architecture.md \
  --model 'qwen38-27b-gpu0-16k:latest' \
  --model 'smtek/Qwen3.8-27B:Q3_K_M' \
  --weights-sha256 9e9aaa43cffbf2606e0e99eec96e8ddb10416e046ca02068e590843a66bc8220
```

Das Manifest enthält eine zufällige Run-ID, je Shard einen Modellnamen, den
gemeinsamen Ollama-Gewichts-Blob-Hash, Prompt-Hash, die sortierte
Dateiliste und für jede Datei SHA-256 plus Shard-Nummer. Manifest und
Repo-Snapshot auf beide GPU-Worker übertragen. Alle Worker brauchen dieselben
Bytes und denselben Graphify-Extraktionsprompt. Die Dateien während des Laufs
nicht ändern.

```bash
python -m graphify.offline_workers run-shard /data/jobs/job.json /data/repo \
  /data/results --shard 0 --endpoint http://127.0.0.1:11436/v1
python -m graphify.offline_workers run-shard /data/jobs/job.json /data/repo \
  /data/results --shard 1 --endpoint http://127.0.0.1:11434/v1
```

Vor der Verarbeitung prüft der Worker über Ollamas `/api/ps`, ob das ganze
Modell im VRAM liegt, und `/api/show`, ob der Gewichts-Blob zum Manifest passt.
Der VRAM wird vor und nach jedem Dateiergebnis erneut geprüft. Externe
VRAM-Änderungen während einer Anfrage kann die Prüfung nicht verhindern.
Ein erneuter Aufruf überspringt validierte Dateiergebnisse.
Unvollständige oder widersprüchliche Ergebnisse stoppen den Lauf. Graphifys
Semantik-Cache speichert außerdem abgeschlossene Chunks lokal.

Nach einer unterbrochenen SSH-Verbindung kann `rsync` denselben Lauf erneut
übertragen. Für jedes Gerät ein eigenes Eingangverzeichnis verwenden, damit
ein Transfer kein bereits angenommenes Ergebnis überschreibt:

```bash
rsync -a --partial --delay-updates gpu-a:/data/results/ /data/incoming/gpu-a/
rsync -a --partial --delay-updates gpu-b:/data/results/ /data/incoming/gpu-b/
python -m graphify.offline_workers import /data/jobs/job.json \
  /data/incoming/gpu-a /data/collected
python -m graphify.offline_workers import /data/jobs/job.json \
  /data/incoming/gpu-b /data/collected
python -m graphify.offline_workers merge /data/jobs/job.json /data/repo \
  /data/collected /data/semantic.json
python -m graphify.offline_workers finalize /data/jobs/job.json /data/repo \
  /data/collected /data/graphify-result
```

`import` prüft Receipt, Run-ID, Datei-/Fragment-Hash, Modell und Prompt. Er
akzeptiert identische Wiederholungen, lehnt abweichende Duplikate ab und
schreibt atomar. `merge` verlangt **alle** erwarteten Dateiergebnisse, prüft
die Quell-Hashes erneut und führt sie in Manifest-Reihenfolge zusammen.
Kollidierende Node-IDs mit unterschiedlichem Inhalt führen zum Fehler. Der
resultierende `semantic.json` ist ein validiertes semantisches Zwischenprodukt.
`finalize` verlangt, dass das Manifest **alle** `.md`/`.txt`-Quellen enthält.
Es schließt alle übrigen erkannten semantischen Dateien per `--exclude` aus,
prüft, dass dabei keine Codequelle verloren geht, und schreibt die Fragmente
in Graphifys per-Datei-Cache, prüft vollständige Cache-Treffer und startet
danach den einmaligen AST-/Build-Lauf auf dem Koordinator. Für unerwartete
LLM-Anfragen ist nur ein unerreichbarer Loopback-Endpunkt gesetzt; ein Cache-
Fehler kann so keinen CPU- oder Netz-Modelllauf auslösen. Das Ergebnis liegt
unter `/data/graphify-result/graphify-out/graph.json`.

SHA-256 schützt die Integrität gegen Übertragungsfehler und versehentliche
Verwechslungen. Es authentifiziert keinen feindlichen Worker. SSH-Schlüssel,
Host-Verifikation und Zugriffsrechte sind dafür weiterhin erforderlich. Das
Ollama-Token/API-Zugangsdaten stehen weder im Manifest noch in Receipts.

Der aktuelle 27B-Lauf nutzt die beiden 16-GB-GPUs von `janpow-ai`. Ein NUC
mit 8 GB VRAM ist dafür zu klein. `run-shard` unterstützt auch einen Shard;
die Ollama-VRAM-Prüfung hängt nicht von NVIDIA ab. Für EVO/ROCm ist das
Verhalten noch nicht geprüft: Vor einem Einsatz müssen Ollama, Modell,
Gewichts-Hash und vollständige VRAM-Belegung dort verifiziert werden.
