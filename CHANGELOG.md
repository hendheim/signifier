# Changelog

Alle nennenswerten Änderungen an *signifier* werden in dieser Datei dokumentiert.
Das Format orientiert sich an [Keep a Changelog](https://keepachangelog.com/de/1.1.0/).

## Gegenüber 0.3.0


### Geändert

- **Begonnene Tag-Listen weiterbearbeiten (beide Tag-Seiten):** Sowohl
  *5.1 Ausdrücke taggen* als auch *4.4.3 Topics taggen* können jetzt eine bereits
  begonnene (teilweise getaggte) Liste laden und fortsetzen. *5.1 Ausdrücke
  taggen* bietet dafür eine Auswahl aus Rohliste und den gespeicherten
  Listen in `resources/stop_pos_tag/` (vorhandene Tags bleiben erhalten);
  *4.4.3 Topics taggen* lädt gespeicherte `*_topic_words_tag_v*.csv` weiter. Die
  Übernahme der Namen in die Document-Topic-Matrix arbeitet dabei
  positionsbasiert, sodass auch schon benannte Topics korrekt übernommen
  werden.
- **Seite 4.4.3 Topics taggen:** Die vergebenen Namen werden
  jetzt nicht nur in die Topic-Word-Matrix, sondern **zusätzlich in die
  Document-Topic-Matrix** geschrieben (deren Topic-Spalten `0,1,…` werden in
  die Namen umbenannt). Der Speicherort ist jetzt der **Topic-Modell-Ordner**
  (statt `resources/topic_names/`); die Default-Dateinamen folgen dem Muster
  `[topic-modell-name]_[matrixtyp]_tag_v1.csv` (z. B.
  `sklearn_lda_10_topic_words_tag_v1.csv` und
  `sklearn_lda_10_document-topics-distribution_tag_v1.csv`).
- **Seite 0 Korpus Verarbeiten (XML-Import):** `<head>`-Elemente innerhalb des `<body>` (TEI-Kapitel-/
  Abschnittsüberschriften) werden nicht mehr in den Fließtext des Korpus
  übernommen; der Text direkt hinter einer Überschrift bleibt erhalten.
  Metadaten aus dem TEI-Header sind davon unberührt.
- **Seiten-Nummerierung:** Durch die neue Seite *Termset erstellen* (7_4.2.3)
  verschieben sich die nachfolgenden Seiten von 7–18 auf 8–19; Inhalte und
  Workflow-Beschriftung bleiben unverändert.
- **`README.md`:** In der Ordnerstruktur sind nun auch die dynamisch erstellten Ordner angeführt und Erläuterungen wurden für weitere Strukturelemente hinzugefügt. 

### Behoben

- **Seite 4.4.3 Topics benennen: Eintragungen wurden erst nach Neuladen übernommen.**
  Die editierte Tabelle wurde bei jedem Rerun in die Eingabe zurück­
  geschrieben, wodurch Editor-Zustand und Eingabe in Konflikt gerieten und
  gerade getippte Namen „sprangen". Die Baseline bleibt jetzt stabil, das
  Ergebnis wird direkt aus dem Editor gelesen – Eintragungen greifen sofort.
- **Seite 5.5 Topics erkunden: Mapping zwischen der Document-Topic-Verteilung und den Metadaten war fehlerhaft.**
  Auf *2. Verarbeitetes Korpus laden* wurde beim Laden eines Topic-Modells die gechunkte Document-Topic-Verteilung auf die Metadaten gemappt. Da die ID der Texte in der Document-Topic-Verteilung um einen Identifier für die Nummer des Chunks pro Dokument erweitert wurde, war das Mapping fehlerhaft. `DISCOVERY_PATTERNS:...{..."topics_dist"[...]}` in `explorer_core/data_store.py` wurde entsprechend angepasst. 
