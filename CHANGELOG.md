# Changelog

Alle nennenswerten Änderungen an *signifier* werden in dieser Datei dokumentiert.
Das Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## Gegenüber 0.2.3

Zwei Schwerpunkte: (1) das Dashboard wurde für große Korpora (getestet bis
~45 Mio. Tokens) beschleunigt, (2) die Topic-/DTTI-Pipeline wurde robuster
gemacht und um eine Seite zum Erstellen von Termsets ergänzt. Die
Analyseergebnisse bleiben unverändert – abgesichert durch Charakterisierungs-
und Paritätstests.

### Hinzugefügt

- **Neue Seite „Termset erstellen" (4.2.3):** Erstellt ein Termset direkt
  aus der Tagset-Pivot-Tabelle der Seite *Tags verarbeiten* — Tags per
  Checkbox markieren (inkl. „Alle markieren"/„Auswahl aufheben"), Vorschau,
  Speichern unter frei wählbarem Namen nach `resources/termsets/`. Das
  gespeicherte Termset wird direkt als aktives Termset übernommen und ist
  mit den Seiten *Termset-Vektoren erkunden* und *DTTI erstellen* kompatibel.
- **Konkordanz (KWIC) erweitert:** Neue Spalte `treffer_autor` (Gesamtzahl
  der Treffer in allen Dokumenten desselben Autors, unabhängig von der
  Treffer-Obergrenze) sowie zwei Optionen: „Nur ganze Wörter (genaue
  Suchzeichenfolge)" und „Groß-/Kleinschreibung beachten" (Standard an).
  Gesucht wird jetzt in der min-Stufe
  (`output/processed_corpus/korpus_min.csv`) mit den Originalwortformen.
- **Token-Index** (`explorer_core/token_index.py`): Das Korpus wird einmal
  pro Prozess tokenisiert und als kompaktes numerisches Array gehalten;
  Kollokationen und Belegdokument-Suche laufen vektorisiert darauf.
- **Automatische Bild-Ablage:** Jede erzeugte Grafik wird zusätzlich zum
  Browser-Download als PNG samt Hyperparameter-TXT unter
  `output/statistics/bilder/` gespeichert (write-once über Inhalts-Hash).
- **Sammel-Download:** Neuer Abschnitt „Ergebnisse herunterladen" auf der
  Seite *Verarbeitetes Korpus laden* — bündelt alle Statistik-CSVs, Bilder
  und Hyperparameter-Dateien in eine ZIP-Datei.
- **Laufzeitanzeige (⏱)** unter den Ergebnissen der Analyse-Seiten.
- **Parquet-Beschleunigung:** Beim ersten Laden der Korpus-CSV entsteht ein
  Parquet-Sidecar; spätere App-Starts laden deutlich schneller (Invalidierung
  über den CSV-Zeitstempel; ohne `pyarrow` sauberer Rückfall auf CSV).

### Geändert

- **Seiten-Nummerierung:** Durch die neue Seite *Termset erstellen* (7_4.2.3)
  verschieben sich die nachfolgenden Seiten von 7–18 auf 8–19; Inhalte und
  Workflow-Beschriftung bleiben unverändert.
- **`-n`-Verarbeitung ohne automatisches Word2Vec:** Bei der
  namensbereinigten Verarbeitung (`--remove-names`) lief das Word2Vec-Training
  (Schritt 10) bisher automatisch mit; es ist jetzt — wie bei der normalen
  Verarbeitung — ausgegliedert und wird über die Seite *Wort-Vektor-Modell
  erstellen* gestartet.
- **Standard-Ersetzungsliste** in `config/signifier_v1.toml` von
  `replacements_dummy.json` auf `replacements_v5.json` umgestellt.

### Performance

- **Token-Statistik-Seite reagiert sofort statt in Minuten:** Einlesen der
  Korpus-Stufen und Tokenzählung liefen bei jeder Widget-Interaktion neu;
  beides ist jetzt gecacht (Invalidierung über Datei-Zeitstempel) und die
  Zählung vektorisiert.
- **Kollokationen ~40–65× schneller pro Abfrage** (4-Mio.-Token-
  Beispielkorpus: ~5 s → ~0,2 s).
- **Caching überlebt Browser-Reload:** CSV-Reads über `st.cache_data`,
  Token-Index über `st.cache_resource`.
- **Ergebnisse bleiben stehen:** Ergebnistabellen der Seite *Ausdrücke
  erkunden* überleben Widget-Interaktionen und Tab-Wechsel.
- **Grafiken rendern nur noch einmal:** Das 300-dpi-PNG wird einmal erzeugt
  und für Anzeige, Download und Ablage wiederverwendet; Streudiagramme auf
  *Texte erkunden* werden nur bei geänderten Daten/Parametern neu gebaut.
- **Dokument-Frequenz** nutzt die vorberechneten Textlängen aus dem
  Token-Index statt das Korpus pro Abfrage neu zu tokenisieren.

### Behoben

- **Jahr-basierte Topic-Dateien wurden nicht erzeugt** (u. a.
  `…_year_topic_matrix.csv` und alle davon abhängigen), sodass Folgeseiten
  mit „No such file or directory" scheiterten. Zwei Ursachen behoben:
  (1) Das Jahr wurde per Regex aus dem formatierten Metadaten-**String**
  gelesen; stand dort kein Jahr, blieb die Tabelle leer und die Datei entfiel.
  Die Jahreszuordnung läuft jetzt **direkt über die Dokument-ID und die
  flexibel bestimmte Jahresspalte** (`year_first`/`year`/`jahr`/`Jahr`/`date`/
  `datum`, robust gegen `1850`, `1850.0`, `1850-03-14`).
  (2) Das Schreiben scheiterte an der **Windows-260-Zeichen-Pfadgrenze** (bei
  tief liegenden Projektordnern); die CSV-Writes nutzen jetzt bei Bedarf
  automatisch den Extended-Length-Pfad (`\\?\`) und melden im Fehlerfall
  vollen Pfad, Pfadlänge und Arbeitsverzeichnis.
- **DTTI-Matrix wurde komplett null bei numerischen Topic-Labels:** Ist der
  Index der Topic-Word-Matrix numerisch (0, 1, 2 …), las pandas ihn als int;
  der Abgleich mit den stets stringbasierten Topic-Spalten der
  Document-Topic-Distribution schlug fehl, sodass jeder Cosinuswert 0 wurde.
  Topic-Index und -Spalten werden nun beidseitig als String geführt.
- **Topic-Postprocessing/DTTI scheiterten an abweichenden Dokument-IDs**
  (häufig `.txt`-Endung in der Verteilung, blanke IDs in Metadaten/DTM). Die
  ID-Zuordnung toleriert jetzt Leerzeichen und `.txt`-Endungen beidseitig
  (tt_s02/s03/s04). Die Logs melden klar, wie viele IDs und Topics zugeordnet
  wurden und wie viele ein Jahr besitzen, und warnen mit Ursachenhinweis.
  *Topics nachverarbeiten* erzeugt zudem garantiert alle angekündigten
  Ausgabedateien (notfalls leer mit Kopfzeile); die Verarbeitungsseiten heben
  [WARN]-Zeilen hervor, *DTTI erkunden* meldet leere Eingaben mit Ursache.
- **Absturz bei „Belegdokumente zu einer Kollokation"** (Seite *Ausdrücke
  erkunden*): Zugriff auf nicht existierende Spalten (`Ziel`/`Kollokat` statt
  `target`/`collocate`) führte zu einem KeyError.
- **Träge Seite *Wort-Vektor-Modell erstellen*:** Die Tokenzählung für die
  Hyperparameter-Voreinstellung (kompletter Korpus-Read) lief bei jeder
  Widget-Interaktion neu; jetzt gecacht.
