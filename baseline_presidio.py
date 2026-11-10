"""
baseline_presidio.py - adapter so evaluate.py can score Microsoft Presidio
(the community-maintained Data Privacy Stack project) on the same benchmark.

Setup once:  python3 -m spacy download en_core_web_lg
"""
from presidio_analyzer import AnalyzerEngine

_ANALYZER = None
LABEL_MAP = {
    "PERSON": "NAME",
    "LOCATION": "LOCATION",
    "EMAIL_ADDRESS": "EMAIL",
    "IP_ADDRESS": "IP",
    # Presidio has no generic API-key recognizer by default, so API_KEY
    # entities will show up as misses unless you add a custom recognizer.
}


def find_spans(text, use_ner=True):
    global _ANALYZER
    if _ANALYZER is None:
        _ANALYZER = AnalyzerEngine()
    results = _ANALYZER.analyze(text=text, language="en", entities=list(LABEL_MAP))
    return sorted((r.start, r.end, LABEL_MAP[r.entity_type]) for r in results)