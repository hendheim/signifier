"""nlp_pipeline – vereinte (Vor-)Verarbeitungs- und Termset/Topic-Pipeline.

Bündelt die früher getrennten Pakete ``fadelive`` (Kern-Schritte s01–s07) und
``procession_termset_topics`` (Termset/Topic-Verarbeitung, Module ``tt_*``).
Die Kern-Schritte stellen jeweils eine ``run``-Funktion bereit und werden über
``pipeline_config.run_pipeline_with_cfg`` orchestriert.
"""
