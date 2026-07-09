# 📖 *signifier* – Dashboard zur Korpusverarbeitung und -exploration

Herausgeber: Hendrick Heimböckel, [ORCID-ID: 0000-0002-4211-9769](https://orcid.org/0000-0002-4211-9769)

## 1. Projektübersicht

*signifier* ist ein Werkzeug zur Verarbeitung von Korpora für **Distant Reading** und **Scalable Reading**. 

Die Funktionen des Dashboards sind Didaktik, Forschung, Reproduzierbarkeit, Nachnutzung und
Weiterentwicklung.

Es umfasst

- die Pipeline, Tools und Ressourcen zur Vorverarbeitung, Verarbeitung Exploration und Visualisierung des Korpus 
- ein Dashboard als grafische Oberfläche.

Die Pipeline und das Dashboard können mit Anpassungen für alle deutschsprachigen Korpora verwendet werden, die sich aus Texten zusammensetzen (wissenschaftlich, journalistisch, politisch, Social-Media, Literatur, Drehbücher etc.). Die Vorverarbeitung ist für Texte optimiert, die keine Paratexte enthalten. 

Anhand der Listen `resources/preprocessing_lists/ocr_post-correction_dictionary.txt`, `resources/preprocessing_lists/replacements.json` und `resources/preprocessing_lists/stopwords.txt` und den Einstellungen in der `config/signifier_v1` können die Ersetzungen sowie Tilgungen an das zugrundeliegende Projekt angepasst werden.

_signifier_ ging aus dem Projekt [*FaDeLive*](https://doi.org/10.5281/zenodo.1797902) hervor, in dem
analoge Quellen fotomechanisch gescannt, mit OCR verarbeitet und in Hinblick auf OCR-Fehler gesäubert wurden. Diese Verarbeitungsstufe TXT (content)* ist die Grundlage für die weitere Prozessierung der Daten. Siehe zur Vorverarbeitung und Verarbeitung des Korpus die Diagramme: <https://doi.org/10.48693/730>.

Alle Funktionen der früheren Versionen wurden in ein **ein gemeinsames, webbasiertes Streamlit-Dashboard** übertragen. 
Änderungen und Erweiterungen sind:
+ Erstellung einer `.csv`-Datenbank auf Grundlage eines Korpus aus `.txt`- oder `.xml`-Dateien,
+ Vorverarbeitung berücksichtigt Groß- und Kleinschreibung
+ Filterung von Named Entities und Verarbeitung des Korpus nach einer Filterung von Named-Entities, `-n`
+ Anpassung der Token-Metadaten-Statistik an die Metadaten des verarbeiteten Korpus
+ semantisches Tagging der POS-getaggten Frequenzliste im Dashboard,
+ die Lemmatisierung erfolgt mit `spaCy` und nicht mit dem `HanoverTagger`,
+ Topic-Modelling mit `scikitlearn` und `MALLET`,
+ Tagging des Topic-Modells
+ Nachverarbeitung des Topic-Modells zur Erzeugung des Document-Term-Topic-Indexes 
+ Metriken für Wort-Vektor- und Topic-Modelle
+ Dimensionsreduktion der Texte mit PCA, MDS und t-SNE und UMAP
+ ein konfigurierbares Metadatenschema,
+ die automatisierte Identifikation von Spaltentrennzeichen.
+ Infoboxen zu Parametern der Verarbeitungsschritte,
+ Speicherung der Parameter.

**Schlagwörter**: Begriffsgeschichte, Computational
Discourse Analyses, Computational Literary Studies, Computerlinguistik, Digital Humanities, Digital Social Sciences, Educational Data Mining, Text Mining, Topic-Modeling, word2vec

<br>

## 2. Installation

Die Installation setzt eine Umgebung mit  **Python 3.11** voraus (idealerweise
3.11.13, für reproduzierbare Word2Vec-Modelle).

Installation der benötigten Bibliotheken: 
```bash
pip install -r requirements.txt
```

Die Verarbeitung erfordert ein spaCy-Modell für Deutsch:
```bash
python -m spacy download de_core_news_lg
```

Die Installation der `MALLET`-Software für das Topic-Modeling ist optional im Tool möglich. Topic-Modeling ist alternativ mit `scikitlearn` möglich, das über `requirements.txt` installiert wird. 

**Start des Dashboards:**

```bash
streamlit run Willkommen.py
```

<br>

## 3. Datengrundlage

Das Dashboard arbeitet mit einer einfachen **Datenbank aus zwei Dateien** im Ordner `korpus/`:

| Datei | Inhalt |
| --- | --- |
| `korpus/korpus.csv` | dieselben Datensätze **plus** den Volltext in der Spalte `content` |
| `korpus/metadaten.csv` | die relevanten bibliografischen Angaben und weitere Metadaten pro Text (eine Zeile je Dokument, *ohne* Volltext) |

**Format**

- **Encoding**: utf-8
- **Trennzeichen**: `,` oder `;` (wird automatisch erkannt)
- **Schlüsselspalte**: `id` (eindeutig) – verknüpft Metadaten und Texte
- **Pflicht im Korpus**: zusätzlich die Spalte `content` (Volltext)

Weitere Metadaten entsprechend des Korpus sind **frei konfigurierbar** (Seite
*Metadatenschema*). Über das Schema wird festgelegt, welche Spalte welche Rolle übernimmt (ID, Jahr,
Anzeige, Facetten).

**Typische Metadaten** (Beispiele, alle optional und benennbar):
`title`, `author_prename`, `author_surname`, `year`, `year_first`, `genre`,
`textclass`, `source`, `editor_surname`, `editor_forname`, `volume`, `edition`, `issue`, `pages`,
`archive`, `address`, `note` u. a.

> Ein Beispieldokument (Marie von Ebner-Eschenbach: *Krambambuli*,
> 1883) wird auf der Startseite des Dashboards als Tabelle gezeigt.

> Ein exemplarisches und umfangreiches Korpus ist etwa das Bildungsromankorpus (Boucher, Herrmann, Hummel, Wiebe 2024), bestehend aus 126 Volltexten: [https://doi.org/10.5281/zenodo.14289199](https://doi.org/10.5281/zenodo.14289199).


<br>

## 4. Datenbank erstellen

In dem Dashboard kann eine `korpus.csv`/`metadaten.csv` aufgebaut werden:

- **aus `.txt`-Dateien** – die Textdateien werden über eine ID-Spalte
  (Dateiname = ID) mit einer Metadaten-CSV zusammengeführt; ohne Metadaten-CSV
  entsteht ein minimales Korpus (`id` + `content`),
- **aus `.xml`-Dateien (TEI)** – Metadaten werden per XPath aus dem TEI-Header
  gelesen, der Fließtext aus `.//tei:text//tei:body`; die Ziel-XML-Pfade sind
  als Metadaten-Kategorien frei definierbar. `<head>`-Elemente im Body
  (Kapitel-/Abschnittsüberschriften) werden dabei **nicht** in den Fließtext
  übernommen; der Text direkt hinter einer Überschrift bleibt erhalten.

Die Metadaten lassen sich vor dem Speichern prüfen und bearbeiten. Geschrieben
werden anschließend `korpus/korpus.csv` und `korpus/metadaten.csv`
(utf-8, Komma-separiert).

<br>

## 5. Aufbau

_signifier_  besteht aus drei Schichten:

1. **NLP-Pipeline** (`s01`–`s07`, `tt_s01`–`tt_s04` ) – die unveränderte Verarbeitungslogik des
   Korpus und die Verarbeitung des Korpus nach der Tilgung von Named Entities.
2. **`explorer_core/`** – UI-freie Logik (Daten/Modelle, Schema, Analysen,
   Topic-Modellierung). Die Module kennen weder Tkinter noch Streamlit und sind
   so auch in Notebooks/Seminaren nutzbar.
3. **Dashboard** (`Willkommen.py` + `pages/`) – die Streamlit-Oberfläche, die
   nur „verkabelt" und auf `explorer_core` zugreift.

<br>

### 5.1 NLP-Pipeline (s01–s07)

Die Pipeline erzeugt aus `korpus/korpus.csv` schrittweise alle Ausgaben unter
`output/`:

1. **`s01_preprocessing`** – Vorverarbeitung → `korpus_min/lem/stop.csv`
    - *min*: Säuberung von OCR-Fehlern und Normalisierung
    - *lem*: Lemmatisierung mit `spaCy`
    - *stop*: Entfernung von Stoppwörtern
2. **`s01_vocabular`** – Vokabular (je Verarbeitungsstufe, optional je Intervall)
4. **`s01_pos_tag`** – POS-Tagging der Top-5000 Ausdrücke mit `spaCy`
5. **`s02_preprocessing_gensim`** – Stufe `korpus_gen` für Wort-Vektor-Modelle
6. **`s03_dtm_tfidf`** – DTM- und TF-IDF-Matrizen
7. **`s04_cosine`** – Kosinus-Matrizen
8. **`s05_dtm_tfidf_cos_intervals`** – Intervall-Matrizen
9. **`s06_tfidf_rank`** – TF-IDF-Ranglisten von Vokabular und Texten

Named Entity Recognition kann optional durchgeführt werden.

10.  **`s08_name_removal`** – Named Entiy Recognition und Verarbeitung des Korpus ohne die named Ausdrücke 

Das Wort-Vektor-Modell setzt die Einstellung weiterer Parameter voraus und steht auf einer separaten Seite.

11. **`s07_word_vector_model`** – Word2Vec-Modelle

Die Pipeline ist über **TOML-Dateien** konfigurierbar.
Voreingestellt ist: `config/signifier_v1.toml`).

**Start der Pipeline** – entweder über die Dashboard-Seite *Korpus verarbeiten*
oder über CLI:

```bash
python run_pipeline.py --project-root . --config config/signifier_v1.toml
# optional nur bestimmte Schritte:
python run_pipeline.py --project-root . --config config/signifier_v1.toml --steps 1 2 3
```

Trennzeichen werden automatisch erkannt; Intervalle und word2vec-Hyperparameter
lassen sich per Argument überschreiben.

<br>

### 5.2 `explorer_core/` – UI-freie Logik

| Modul | Funktion |
| --- | --- |
| `data_store.py` | Laden/Cachen aller Datenquellen und Modelle (`DataStore`, `ModelStore`) |
| `schema.py` | konfigurierbares Metadatenschema mit Auto-Detection-Fallback |
| `corpus_build.py` | Korpus-Datenbank aus `.txt` oder `.xml`-Dateien zusammenstellen |
| `corpus_segment.py` | lange Texte in Chunks zerlegen (für Topic-Modelle) |
| `analysis_terms.py` | Frequenz, TF-IDF-Rang, Dokument-Frequenz, Konkordanz, Kollokationen, Wortverläufe |
| `analysis_texts.py` | Streudiagramme/Projektionen der Texte (PCA, MDS, t-SNE, UMAP) |
| `analysis_topics.py` | Topicverläufe und Tag-Topic-Analysen |
| `analysis_stats.py` | Statistik: Tokens je Stufe, Milestones, Frequenz je Metadatenspalte |
| `analysis_vectors.py` | ähnlichste Wörter, Embedding-Vergleich, Netzwerk, Termset-Cluster/Wortwolke/Dendrogramme |
| `topic_model.py` | Topic-Modellierung mit scikit-learn (NMF/LDA) |
| `topic_metrics.py` | Qualitätsmetriken für Topic-Modelle (Diversität, Kohärenz) |
| `wvm_metrics.py` | Qualitätsmetriken für das Word2Vec-Modell |
| `mallet_runner.py` / `mallet_setup.py` | MALLET einrichten und als Topic-Modeller aufrufen |
| `tagging.py` / `tag_processing.py` | semantisches Taggen der POS-Liste und deren Weiterverarbeitung; bereits begonnene Tag-Listen können erneut geladen und fortgesetzt werden |
| `topic_tagging.py` | Topics einer vollständigen Topic-Word-Matrix benennen; die Namen werden versioniert in den **Topic-Modell-Ordner** geschrieben – in die Topic-Word-Matrix **und** die Document-Topic-Matrix (Dateimuster `[modell]_[matrixtyp]_tag_v1.csv`) |
| `pipeline_runner.py` | startet die NLP-Pipeline als Subprozess (Live-Log) |
| `viz_export.py` | speichert zu jeder Grafik die verwendeten Hyperparameter mit |

<br>

### 5.3 Dashboard-Seiten

Die Seiten erscheinen links im Menü in dieser Reihenfolge (Dateipräfix steuert die Sortierung):

**Erstellung des Korpus, Vorverarbeitung und Einrichtung**

- **0 · Korpus Datenbank erstellen** – Datenbank aus `txt-`/`xml`-Dateien bauen
  
-  **1 · Korpus verarbeiten** – NLP-Pipeline (`s01`–`s06`) starten; zusätzlich Button *„Namen aus dem Vokabular tilgen"* → erzeugt die namensbereinigte Variante **`-n`** (Personennamen/Autoren getilgt, Orte erhalten; ab `corpus_stop`, Ausgaben in `…-n`-Ordnern)

- **2 · Verarbeitetes Korpus laden** – Projektordner setzen und Datenpfade prüfen
  
- **3 · Metadatenschema fixieren** – Rollen der Metadatenspalten festlegen (`config/metadata_schema.yaml`)


**Verarbeitung**
  
- **4.1 · Token-Statistik erstellen** – Tokens je Stufe, Milestones (Token-Abschnitte mit Jahresspanne), Frequenz je Metadatenspalte (Dokumente + Tokens je Wert); per Checkbox auswählbar

- **4.2.1 · Ausdrücke taggen** – POS-Frequenzliste um die Tags `tag1, tag2, tag3` ergänzen; bereits begonnene Tag-Listen aus `resources/stop_pos_tag/` können geladen und weiterbearbeitet werden
  
- **4.2.2 · Tags verarbeiten** – getaggte POS-Liste verarbeiten (`tt01`)

- **4.2.3 · Termset erstellen** – aus der Tags-Pivot-Tabelle (Seite *Tags verarbeiten*) per Markierung ein Termset zusammenstellen und nach `resources/termsets/` speichern
  
- **4.3 · Wort-Vektor-Modelle erstellen** – ein Wort-Vektor-Modell wird erstellt (`s07`)
  
- **4.4.1 · MALLET einrichten** – Java prüfen, die Topic-Modellierungs-Software MALLET herunterladen/entpacken
  
- **4.4.2 · Topic-Modell erstellen** – Topics mit scikit-learn (NMF/LDA) bzw. MALLET, optional mit Chunking
  
- **4.4.3 · Topics taggen** – vollständige Topic-Word-Matrix ansehen und je Topic einen Namen vergeben; die Namen werden in den **Topic-Modell-Ordner** geschrieben – in die Topic-Word-Matrix **und** die Document-Topic-Matrix (deren Topic-Spalten umbenannt werden); begonnene Listen lassen sich fortsetzen
  
- **4.4.4 · Topics nachverarbeiten** – Topic-Ranking (`tt02`)
  
- **4.4.5 · Document-Term-Topic-Index erstellen** –Termset-Topic-Text-Verhältnisse berechnen und nachverarbeiten (`tt03`–`tt04`)


**Exploration**

- **5.1 · Ausdrücke erkunden** – Frequenz, TF-IDF-Rang, Dokument-Frequenz, Konkordanz (KWIC), Kollokationen (FREQ/PMI), Wortverläufe
  
- **5.2 · Texte erkunden** – Streudiagramme (PCA, MDS, t-SNE, UMAP) und Dendrogramme der Text-Ähnlichkeit
  
- **5.3 · Wort-Vektoren erkunden** – Embeddings, Embedding-Vergleich, semantisches Netzwerk, Modell-Metriken
  
- **5.4 · Termset-Vektoren erkunden** – UMAP-Cluster, Wortwolke und Dendrogramme eines Termsets
  
- **5.5 · Topics erkunden** – diachrone Topicverläufe ausgewählter Topics
  
- **5.6 · Tag-Topics erkunden** – Tag-Topic-Relevanz (Bubbles), Jahresverteilungen, Tendenzkurven, Ranglisten
  

<br>

## 6. Funktionen der Tag-, Termset- und Topic-Verarbeitung

_Die Tag-Topic-Verhältnisse zu visualisieren bietet sich nur bei größeren Korpora an._

Eines der grundlegenden Ziele des übergeordneten Projektes war die Ermittlung von Ausdrücken, die die Fachlichkeit der Literaturvermittlung im 19. Jahrhundert bestimmen. Hierzu wurden zwei Verfahren zunächst separat angewendet, dann kombiniert: qualitatives, semantisches Taggen sowie algorithmische Modellierung von Topics und das semantische Taggen der Topics. 

Mit der Verarbeitung in _4.4.4 · Topics nachverarbeiten_ und _4.4.5 · Document-Term-Topic-Index erstellen_ werden kontrollierte Vokabulare und Topic-Modelle aufeinander abgebildet. Mit _6.6 · DTTI erkunden_ werden sie visualisiert und erkundet. 

**Exemplarische Anwendung**:

- eine POS-getaggte Vokabelliste semantisch taggen, indem die Spalten `tag1`,
  `tag2` und `tag3` ergänzt werden, die Ausdrücke mit bis zu drei Kategorien
  abstrahieren (_4.2.1 · Ausdrücke taggen_),
- daraus eine Pivot-Tabelle erzeugen (_4.2.2 · Tags verarbeiten_),
- für die Forschungsfrage relevante Tags und Ausdrücke aus der Tabelle wählen und ein Termset bilden,
- Termsets auf Topics und Texte abbilden und einen Dokument-Term-Topic-Index berechnen (_4.4.4 · Topics nachverarbeiten_, _4.4.5 · Document-Term-Topic-Index erstellen_).

Für die Modellierung der Topics kann das Dashboard `scikit-learn` (NMF/LDA) oder
`MALLET` nutzen.

<br>

## 7. Bibliotheken

Siehe `requirements.txt`. 

## 8. Projektstruktur

```
signifier/
|   CHANGELOG.md
|   CITATION.cff
|   LICENSE.md
|   README.md
|   requirements.txt
|   run_pipeline.py                     # Startet die NLP-Pipeline über CLI
|   start_dashboard.bat
|   start_dashboard.sh
|   start_pipeline.bat
|   start_pipeline.sh
|   ui_helpers.py                       # Streamlit-Verkabelung (Session, Downloads, Fehler)
|   Willkommen.py                       # Streamlit-Startseite
|               
+---.streamlit
|       config.toml
|
+---config
|       metadata_schema.yaml
|       signifier_v1.toml
|
+---explorer_core                        # UI-freie Logik
|       analysis_stats.py                # Tokenstatistik im Verhältnis zu den Metadaten
|       analysis_terms.py                # Frequenzen, Kollokationen, Konkordanzen, Wortverläufe
|       analysis_texts.py                # Dimensionsreduktion und Cluster
|       analysis_topics.py               # Topicverläufe
|       analysis_vectors.py              # Word-Embeddings
|       corpus_build.py                  # Korpus-Erstellung aus .txt- oder .xml-Dateien
|       corpus_segment.py                # Chunking für die Erstellung von Topic-Modellen
|       data_store.py                    # Korpus anhand von verarbeiteten Dateien laden
|       mallet_runner.py                 # MALLET ausführen
|       mallet_setup.py                  # MALLET laden 
|       pipeline_runner.py               # Orchestrierung der NLP-Pipeline
|       schema.py                        # Flexibilisierung der Metadaten
|       tagging.py                       # manuelles Tagging von Ausdrücken
|       tag_processing.py                # Verarbeitung der Tags
|       token_index.py                   # Caching der Tokens
|       topic_metrics.py                 # einfache Metriken für das Topic-Modell
|       topic_model.py                   # Erstellung von Topic-Modellen
|       topic_tagging.py                 # manuelles Tagging von Topic-Modellen
|       viz_export.py                    # Unterstützungsfunktionen zum Rendern der Grafiken
|       wvm_metrics.py                   # einfache Metriken für Wort-Vektor-Modelle
|       __init__.py
|
+---korpus/                             # [Dynamic] wird manuell oder über Dateiimport erstellt
|       korpus.csv                      # Metadaten + Content
|       metadaten.csv                   # Nur Metadaten
|
+---nlp_pipeline                        # Voverrabeitung, NER, Tag-Topic-Verarbeitung
|       pipeline_config.py
|       pipeline_utils.py
|       s01_pos_tag.py
|       s01_preprocessing.py
|       s01_vocabulary.py
|       s02_gensim_preprocessing.py
|       s03_dtm_tfidf.py
|       s04_cosine.py
|       s05_cosine_intervals.py
|       s06_tfidf_rank.py
|       s07_word_vectors.py
|       s08_name_removal.py
|       tt_s01_stop_pos_tag.py
|       tt_s02_topics.py
|       tt_s03_dtti.py
|       tt_s04_dtti.py
|       __init__.py
|
+---output/                             # [Dynamic] Erzeugte Daten
|       processed_corpus/               # korpus_min/lem/stop.csv
|       vocabular/                      # Vokabular und Frequenzen
|       statistics/                     # Token-Statistiken im Verhältnis zu den Metadaten
|       dtm_tfidf_stop/                 # DTM- und TF-IDF-Matrizen
|       cosine/                         # Kosinus-Matrizen
|       intervals/                      # Intervall-spezifische Ausgaben
|       tfidf_rank/                     # TF-IDF-Rankings
|       word2vec_models/                # Wort-Vektor-Modell
|       processed_tag/                  # Verarbeitete Tags
|       processed_topics/               # Je Modell: <Modell>/
|       processed_termset/              # Je Termset: <Termset>/<Modell>/

+---pages                               # Dashboard-Seiten
|       0_0_Korpus_Datenbank_erstellen.py
|       10_4.4.2_Topic-Modell_erstellen.py
|       11_4.4.3_Topics_taggen.py
|       12_4.4.4_Topics_nachverarbeiten.py
|       13_4.4.5_Document-Term-Topic-Index_erstellen.py
|       14_5.1_Ausdrücke_erkunden.py
|       15_5.2_Texte_erkunden.py
|       16_5.3_Wort-Vektoren_erkunden.py
|       17_5.4_Termset-Vektoren_erkunden.py
|       18_5.5_Topics_erkunden.py
|       19_5.6_DTTI erkunden.py
|       1_1._Korpus_verarbeiten.py
|       2_2._Verarbeitetes_Korpus_laden.py
|       3_3._Metadatenschema_fixieren.py
|       4_4.1_Token-Statistik_erstellen.py
|       5_4.2.1_Ausdrücke_taggen.py
|       6_4.2.2_Tags_verarbeiten.py
|       7_4.2.3_Termset_erstellen.py
|       8_4.3_Wort-Vektor-Modell_erstellen.py
|       9_4.4.1_MALLET_einrichten.py
|
\---resources
    +---preprocessing_lists             # Stoppwörter, Ersetzungen, OCR
            ocr_post-correction_dictionary.txt
            ocr_post-correction_dictionary_dummy.txt
            replacements_dummy.json
            replacements_v5.json
            stopwords_dummy.txt
            stopwords_v4.txt
    +---stop_pos_tag/                   # [Dynamic] Getaggte POS-Listen
    +---termsets/                       # [Dynamic] Kontrollierte Vokabulare
    +---mallet/                         # [Dynamic] MALLET-Starter-Pfad
    \---topic-models/                   # [Dynamic] Topic-Modell-Dateien
```

## 8. Verwendete KI
_signifier_ wurde mit Fable 5 und Opus 4.8 von Anthropic geschrieben. 
Unsagbares Staunen. 
