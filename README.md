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

Anhand der Listen `resources/ocr_post-correction_dictionary.txt`, `resources/replacements.json` und `resources/stopwords.txt` und den Einstellungen in der `config/signifier_v1` können die Ersetzungen sowie Tilgungen an das zugrundeliegende Projekt angepasst werden.

_signifier_ ging aus dem Projekt [*FaDeLive*](https://doi.org/10.5281/zenodo.1797902](https://doi.org/10.5281/zenodo.17979024) hervor, in dem
analoge Quellen fotomechanisch gescannt, mit OCR verarbeitet und in Hinblick auf OCR-Fehler gesäubert wurden. Diese Verarbeitungsstufe TXT (content)* ist die Grundlage für die weitere Prozessierung der Daten. Siehe zur Vorverarbeitung und Verarbeitung des Korpus die Diagramme: <https://doi.org/10.48693/730>

Alle Funktionen der früheren Versionen wurden in ein **ein gemeinsames, webbasiertes Streamlit-Dashboard** übertragen. 
Änderungen und Erweiterungen sind:
+ die Erstellung einer `.csv`-Datenbank auf Grundlage eines Korpus aus `.txt`- oder `.xml`-Dateien,
+ semantisches Tagging der POS-getaggten Frequenzliste im Dahsboard,
+ die Lemmatisierung erfolgt mit `spaCy` und nicht mit dem `HanoverTagger`,
+ Topic-Modelling mit `scikitlearn` und `MALLET`,
+ Metriken für Wort-Vektor- und Topic-Modelle
+ die Dimensionsreduktion der Texte mit PCA, MDS und t-SNE,
+ ein frei konfigurierbares Metadatenschema,
+ die automatisierte Identifikation von Spaltentrennzeichen.
+ Infoboxen zu Parametern der Verarbeitungsschritte,
+ Speicherung der Parameter.

**Schlagwörter**: Computerlinguistik, Didaktik des Deutschunterrichts, Computational Literary Studies,
Literaturunterricht, Historische Bildungsforschung, Text Mining, Computational
Discourse Analyses

<br>

## 2. Installation

Die Installation setzt eine Umgebung mit **Python 3.11** voraus (idealerweise
3.11.13, für reproduzierbare Word2Vec-Modelle).

```bash
pip install -r requirements.txt
# Deutsches spaCy-Modell (falls nicht über requirements gezogen):
python -m spacy download de_core_news_lg
```

Hinweise:

- **MALLET** ist optional (Java-Programm, kein pip-Paket) und kann über die
  Seite *MALLET einrichten* halb-automatisch installiert werden.

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

Welche weiteren Metadatenspalten es gibt, ist **frei konfigurierbar** (Seite
*Metadatenschema*). Über das Schema wird festgelegt, welche Spalte welche Rolle übernimmt (ID, Jahr,
Anzeige, Facetten). Fehlt eine Angabe, greifen dieselben Heuristiken wie bisher,
sodass bestehende Korpora ohne Änderung funktionieren.

**Typische Metadaten** (Beispiele, alle optional und benennbar):
`title`, `author_prename`, `author_surname`, `year`, `year_first`, `genre`,
`textclass`, `source`, `editor_surname`, `editor_forname`, `volume`, `edition`, `issue`, `pages`,
`archive`, `address`, `note` u. a.

> Ein Beispieldokument (Marie von Ebner-Eschenbach: *Krambambuli*,
> 1883) wird auf der Startseite des Dashboards als Tabelle gezeigt.

> Ein exemplarisches und umfangreiches Korpus wäre etwa der Bildungsromankorpus (Boucher, Herrmann, Hummel, Wiebe 2024), bestehend aus 126 Volltexten: [https://doi.org/10.5281/zenodo.14289199](https://doi.org/10.5281/zenodo.14289199).


<br>

## 4. Datenbank erstellen

In dem Dashboard kann eine `korpus.csv`/`metadaten.csv` aufgebaut werden (Seite *Korpus Datenbank erstellen*):

- **aus `.txt`-Dateien** – die Textdateien werden über eine ID-Spalte
  (Dateiname = ID) mit einer Metadaten-CSV zusammengeführt; ohne Metadaten-CSV
  entsteht ein minimales Korpus (`id` + `content`),
- **aus `.xml`-Dateien (TEI)** – Metadaten werden per XPath aus dem TEI-Header
  gelesen, der Fließtext aus `.//tei:text//tei:body`; die Ziel-XML-Pfade sind
  als Metadaten-Kategorien frei definierbar.

Die Metadaten lassen sich vor dem Speichern prüfen und bearbeiten. Geschrieben
werden anschließend `korpus/korpus.csv` und `korpus/metadaten.csv`
(utf-8, Komma-separiert).

<br>

## 5. Aufbau

_signifier_  besteht aus drei Schichten:

1. **NLP-Pipeline** (`s01`–`s07`, `tt_s01`–`tt_s04` ) – die unveränderte Verarbeitungslogik des
   Korpus.
2. **`explorer_core/`** – UI-freie Logik (Daten/Modelle, Schema, Analysen,
   Topic-Modellierung). Die Module kennen weder Tkinter noch Streamlit und sind
   so auch in Notebooks/Seminaren nutzbar.
3. **Dashboard** (`Willkommen.py` + `pages/`) – die Streamlit-Oberfläche, die
   nur „verkabelt" und auf `explorer_core` zugreift.

<br>

### 5.1 NLP-Pipeline (s01–s07)

Die Pipeline erzeugt aus `korpus/korpus.csv` schrittweise alle Ausgaben unter
`output/`:

1. **`s01_1_preprocessing`** – Vorverarbeitung → `korpus_min/lem/stop.csv`
    - *min*: Säuberung von OCR-Fehlern und Normalisierung
    - *lem*: Lemmatisierung mit `spaCy`
    - *stop*: Entfernung von Stoppwörtern
2. **`s01_2_vocabular`** – Vokabular (je Verarbeitungsstufe, optional je Intervall)
3. **`s01_3_statistics`** – Statistiken zu Tokens (sind deutlich ausbaufähig)
4. **`s01_4_pos_tag`** – POS-Tagging der Top-5000 Ausdrücke mit `spaCy`
5. **`s02_preprocessing_gensim`** – Stufe `korpus_gen` für Wort-Vektor-Modelle
6. **`s03_dtm_tfidf`** – DTM- und TF-IDF-Matrizen
7. **`s04_cosine`** – Kosinus-Matrizen
8. **`s05_dtm_tfidf_cos_intervals`** – Intervall-Matrizen
9. **`s06_tfidf_rank`** – TF-IDF-Ranglisten von Vokabular und Texten
10. **`s07_word_vector_model`** – Word2Vec-Modelle

Die Pipeline ist über **TOML-Dateien** konfigurierbar
(`config/signifier_v1.toml`).

**Start der Pipeline** – entweder über die Dashboard-Seite *Korpus verarbeiten*
oder über CLI:

```bash
python run_pipeline.py --project-root . --config config/fadelive_v3.toml
# optional nur bestimmte Schritte:
python run_pipeline.py --project-root . --config config/fadelive_v3.toml --steps 1 2 3
```

Trennzeichen werden automatisch erkannt; Intervalle und Word2Vec-Hyperparameter
lassen sich per Argument überschreiben.

<br>

### 5.2 `explorer_core/` – UI-freie Logik

| Modul | Aufgabe |
| --- | --- |
| `data_store.py` | Laden/Cachen aller Datenquellen und Modelle (`DataStore`, `ModelStore`) |
| `schema.py` | konfigurierbares Metadatenschema mit Auto-Detection-Fallback |
| `corpus_build.py` | Korpus-Datenbank aus `.txt` oder `.xml`-Dateien zusammenstellen |
| `corpus_segment.py` | lange Texte in Segmente zerlegen (für Topic-Modelle) |
| `analysis_terms.py` | Frequenz, TF-IDF-Rang, Dokument-Frequenz, Konkordanz, Kollokationen, Wortverläufe |
| `analysis_texts.py` | Streudiagramme/Projektionen der Texte (PCA, MDS, t-SNE, UMAP) |
| `analysis_topics.py` | Topicverläufe und Tag-Topic-Analysen |
| `analysis_vectors.py` | ähnlichste Wörter, Embedding-Vergleich, Netzwerk, Termset-Cluster/Wortwolke/Dendrogramme |
| `topic_model.py` | Topic-Modellierung mit scikit-learn (NMF/LDA) |
| `topic_metrics.py` | Qualitätsmetriken für Topic-Modelle (Diversität, Kohärenz) |
| `wvm_metrics.py` | Qualitätsmetriken für das Word2Vec-Modell |
| `mallet_runner.py` / `mallet_setup.py` | MALLET einrichten und als Topic-Modeller aufrufen |
| `tagging.py` / `tag_processing.py` | semantisches Taggen der POS-Liste und deren Weiterverarbeitung|
| `pipeline_runner.py` | startet die NLP-Pipeline als Subprozess (Live-Log) |
| `viz_export.py` | speichert zu jeder Grafik die verwendeten Hyperparameter mit |

<br>

### 5.3 Dashboard-Seiten

Die Seiten erscheinen links im Menü in dieser Reihenfolge (Dateipräfix steuert die Sortierung):

**Verarbeitung & Einrichtung**

**0 · Korpus Datenbank erstellen** – Datenbank aus `txt-`/`xml`-Dateien bauen
**1 · Korpus verarbeiten** – NLP-Pipeline (`s01`–`s06`) starten
**2 · Semantisches Taggen** – POS-Frequenzliste um die Tags `tag1, tag2, tag3` ergänzen
**3 · Semantische Tags verarbeiten** – getaggte POS-Liste verarbeiten (`tt01`)
**4 · Wort-Vektor-Modelle erstellen** – ein Wort-Vektor-Modell wird erstellt (`s07`)
**5 · MALLET einrichten** – Java prüfen, die Topic-Modellierungs-Software MALLET herunterladen/entpacken
**6 · Topic-Modell erstellen** – Topics mit scikit-learn (NMF/LDA) bzw. MALLET, optional mit Chunking
**7 · Topics nachverarbeiten** – Topic-Ranking (`tt02`)
**8 · Document-Term-Topic-Index** –Termset-Topic-Text-Verhältnisse berechnen und nachverarbeiten (`tt03`–`tt04`)
**9 · Verarbeitetes Korpus laden** – Projektordner setzen und Datenpfade prüfen
**10 · Metadatenschema** – Rollen der Metadatenspalten festlegen (`config/metadata_schema.yaml`)

**Exploration**

**11 · Ausdrücke** – Frequenz, TF-IDF-Rang, Dokument-Frequenz, Konkordanz (KWIC), Kollokationen (FREQ/PMI), Wortverläufe
**12 · Texte** – Streudiagramme (PCA, MDS, t-SNE, UMAP) und Dendrogramme der Text-Ähnlichkeit
**13 · Wort-Vektoren** – Embeddings, Embedding-Vergleich, semantisches Netzwerk, Modell-Metriken
**14 · Termset-Vektoren** – UMAP-Cluster, Wortwolke und Dendrogramme eines Termsets
**15 · Topics** – diachrone Topicverläufe ausgewählter Topics
**16 · Tag-Topics** – Tag-Topic-Relevanz (Bubbles), Jahresverteilungen, Tendenzkurven, Ranglisten

<br>

## 6. Funktionen der Tag-, Termset- und Topic-Verarbeitung

_Die Tag-Topic-Verhältnisse zu visualisieren bietet sich nur bei größeren Korpora an. Die Voreinstellung der entsprechenden Dateien sowie die Verarbeitung Visualisierung der entsprechenden Dateien ist noch fehlerhaft._

Eines der grundlegenden Ziele des übergeordneten Projektes war die Ermittlung von Ausdrücken, die die Fachlichkeit der Literaturvermittlung im 19. Jahrhundert bestimmen. Hierzu wurden zwei Verfahren zunächst separat angewendet, dann kombiniert: qualitatives, semantisches Taggen sowie algorithmische Modellierung von Topics und das semantische Taggen der Topics. 

Mit der Verarbeitung in _7 · Topics nachverarbeiten_ und _8 · Document-Term-Topic-Index_ werden kontrollierte Vokabulare und Topic-Modelle aufeinander abgebildet. Mit _16 · Tag-Topics_ werden sie visualisiert und erkundet. 

**Exemplarische Anwendung**:

- eine POS-getaggte Vokabelliste semantisch taggen, indem die Spalten `tag1`,
  `tag2` und `tag3` ergänzt werden, die Ausdrücke mit bis zu drei Kategorien
  abstrahieren (_2 · Semantisches Taggen_),
- daraus eine Pivot-Tabelle erzeugen (_3 · Semantische Tags verarbeiten_),
- für die Forschungsfrage relevante Tags und Ausdrücke aus der Tabelle wählen und ein Termset bilden,
- Termsets auf Topics und Texte abbilden und einen Dokument-Term-Topic-Index berechnen (_7 · Topics nachverarbeiten_, _8 · Document-Term-Topic-Index_).

Für die Modellierung der Topics kann das Dashboard `scikit-learn` (NMF/LDA) oder
`MALLET` nutzen; alternativ lässt sich die ohtm-Pipeline von Bayerschmidt
verwenden: <https://github.com/bayerschphi/ohtm_pipeline>.

<br>

## 7. Bibliotheken

Siehe `requirements.txt`. 

## 8. Projektstruktur

```
signifier
│   Willkommen.py            				# Streamlit-Startseite
│   ui_helpers.py            				# Streamlit-Verkabelung (Session, Downloads, Fehler)
│   run_pipeline.py          				# startet die NLP-Pipeline über CLI
│   requirements.txt
│
├───pages                   				# Dashboard-Seiten (0_… bis 12_…)
│       0_Korpus_Datenbank_erstellen.py		
│       1_Korpus_verarbeiten.py
│       2_Semantisches_Taggen.py
│       3_Semantische_Tags_verarbeiten.py
│       4_Wort-Vvektor-Modell_erstellen
│       5_MALLET_einrichten.py
│       6_Topic-Modell_erstellen.py
│       7_Topics_nachbearbeiten.py
│       8_Document-Term-Topic-Index.py
│       9_Verarbeitetes_Korpus_laden.py
│       10_Metadatenschema.py
│       11_Ausdrücke.py
│       12_Texte.py
│       13_Wort-Vektoren.py
│       14_Termset-Vektoren.py
│       15_Topics.py
│       16_Tag_Topics.py
│
├───explorer_core           				# UI-freie Logik
│       data_store.py
│       schema.py
│       corpus_build.py
│       corpus_segment.py
│       analysis_terms.py
│       analysis_texts.py
│       analysis_topics.py
│       analysis_vectors.py
│       topic_model.py
│       topic_metrics.py
│       wvm_metrics.py
│       mallet_runner.py
│       mallet_setup.py
│       tagging.py
│       tag_processing.py
│       pipeline_runner.py
│       viz_export.py
│
├───nlp_pipeline
│       pipeline_config.py
│       pipeline_utils.py
│       s01_1_preprocessing.py … s07_word_vector_model.py
│       tt_s01_stop_pos_tag.py       	
│       tt_s02_topics.py
│       tt_s03_dtti.py
│       tt_s04_dtti.py
│
├───config
│       signifier_v1.toml
│       metadata_schema.yaml
│
├───korpus
│       korpus.csv          				# Metadaten + content
│       metadaten.csv        				# nur Metadaten
│
├───resources
│   │   ocr_post-correction_dictionary_dummy.txt
│   │   ocr_post-correction_dictionary.txt
│   │   replacements_dummy.json
│   │   replacements_v3.json
│   │   stopwords_v3.txt
│   ├───stop_pos_tag         				# getaggte POS-Listen
│   ├───termsets             				# kontrollierte Vokabulare (Pivot-Tabellen)
│   ├───mallet              				# MALLET-Starter-Pfad
│   └───topic-models        				# Topic-Modelle (sklearn/MALLET)
│
└───output                 					# erzeugte Daten (Pipeline + Verarbeitung)
    ├───processed_corpus	   				# korpus_min/lem/stop.csv
    ├───vocabular 			 				# Vokabular und Frequenzen sowie das POS-getaggte Vokabular 
    ├───statistics           
    ├───dtm_tfidf_stop       				# DTM- und TF-IDF-Matrizen
    ├───cosine 				 				# Kosinus-Matrizen
    ├───intervals 		     				# intervall-spezifische Ausgaben
    ├───tfidf_rank 							# ranking der Ausdrücke und Texte anhand der TF-IDF-Werte
    ├───word2vec_models 					# Wort-Vektor-Modell
    ├───processed_tag 						# verarbeitete Tags
    ├───processed_topics 					# verarbeitete Topics
    └───processed_termset 					# verarbeitete Termsets (Document-Term-Topic-Index)
```

## 8. Verwendete KI
_signifier_ wurde mit Fable 5 und Opus 4.8 von Anthropic geschrieben. 
Das Staunen ist unaussprechlich. 