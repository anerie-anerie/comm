"""
scrubber.py - a local, no-cost PII scrubber for LLM prompts.

Layer 1: regex for structured PII (emails, IPs, API keys / tokens)
Layer 2: entropy check for generic secret-looking strings
Layer 3: a small local NER model (Hugging Face) for names and locations

Usage:
    python3 scrubber.py "Hi, I'm Anerie Patel (anerie@uwaterloo.ca) from Waterloo"
    echo "some text" | python3 scrubber.py
    python3 scrubber.py --no-ner "text"        # regex only (fast, no model download)
"""
import argparse
import math
import re
import sys

# ---------------------------------------------------------------- regex layer
# Lookarounds are used instead of \b so keys ending in "-" or "_" still match.
REGEX_PATTERNS = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+"),
    "API_KEY": re.compile(
        r"(?<![A-Za-z0-9_-])(?:"
        r"sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}"      # OpenAI / Anthropic style
        r"|AKIA[0-9A-Z]{16}"                         # AWS access key id
        r"|gh[pousr]_[A-Za-z0-9]{36,}"               # GitHub tokens
        r"|xox[abprs]-[A-Za-z0-9-]{10,}"             # Slack tokens
        r"|AIza[0-9A-Za-z_-]{35}"                    # Google API key
        r")(?![A-Za-z0-9_-])"
    ),
    "IP": re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
}

# generic "looks like a secret" candidates: long, mixed letters+digits, high entropy
GENERIC_CANDIDATE = re.compile(r"(?<![A-Za-z0-9+/_=-])[A-Za-z0-9+/_=-]{24,}(?![A-Za-z0-9+/_=-])")

# priority: lower number wins when spans overlap
PRIORITY = {"EMAIL": 0, "API_KEY": 0, "IP": 0, "GENERIC_KEY": 1, "NAME": 2, "LOCATION": 2}


def _valid_ip(s):
    return all(0 <= int(part) <= 255 for part in s.split("."))


def _entropy(s):
    counts = {c: s.count(c) for c in set(s)}
    return -sum((n / len(s)) * math.log2(n / len(s)) for n in counts.values())


def regex_spans(text):
    spans = []
    for label, pat in REGEX_PATTERNS.items():
        for m in pat.finditer(text):
            if label == "IP" and not _valid_ip(m.group()):
                continue
            spans.append((m.start(), m.end(), label))
    for m in GENERIC_CANDIDATE.finditer(text):
        s = m.group()
        if re.search(r"\d", s) and re.search(r"[A-Za-z]", s) and _entropy(s) >= 3.5:
            spans.append((m.start(), m.end(), "API_KEY"))  # reported as API_KEY
    return spans


# ------------------------------------------------------------------ NER layer
_NER = None
NER_MODEL = "dslim/bert-base-NER"
NER_LABEL_MAP = {"PER": "NAME", "LOC": "LOCATION"}


def _get_ner():
    global _NER
    if _NER is None:
        from transformers import pipeline  # imported lazily so --no-ner is fast
        _NER = pipeline("ner", model=NER_MODEL, aggregation_strategy="simple")
    return _NER


def ner_spans(text, min_score=0.80):
    spans = []
    for e in _get_ner()(text):
        label = NER_LABEL_MAP.get(e["entity_group"])
        if label and e["score"] >= min_score:
            spans.append((e["start"], e["end"], label))
    return spans


# ------------------------------------------------------------ merge + replace
def _resolve_overlaps(spans):
    """Keep non-overlapping spans. Regex beats NER; longer beats shorter."""
    ordered = sorted(spans, key=lambda s: (PRIORITY.get(s[2], 3), -(s[1] - s[0])))
    kept = []
    for start, end, label in ordered:
        if all(end <= ks or start >= ke for ks, ke, _ in kept):
            kept.append((start, end, label))
    return sorted(kept)


def find_spans(text, use_ner=True):
    spans = regex_spans(text)
    if use_ner:
        spans += ner_spans(text)
    return _resolve_overlaps(spans)


def scrub(text, use_ner=True):
    """Return (clean_text, mapping). mapping: placeholder -> original value."""
    spans = find_spans(text, use_ner)
    counters, seen, mapping = {}, {}, {}
    pieces, cursor = [], 0
    for start, end, label in spans:
        value = text[start:end]
        key = (label, value)
        if key not in seen:
            counters[label] = counters.get(label, 0) + 1
            seen[key] = f"[{label}_{counters[label]}]"
            mapping[seen[key]] = value
        pieces.append(text[cursor:start])
        pieces.append(seen[key])
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), mapping


def restore(text, mapping):
    """Put originals back into an LLM reply that reuses the placeholders."""
    for placeholder, original in mapping.items():
        text = text.replace(placeholder, original)
    return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Scrub PII from a prompt.")
    ap.add_argument("text", nargs="?", help="prompt text (or pipe via stdin)")
    ap.add_argument("--no-ner", action="store_true", help="regex only")
    ap.add_argument("--show-map", action="store_true", help="print placeholder map")
    args = ap.parse_args()
    raw = args.text if args.text is not None else sys.stdin.read()
    clean, mapping = scrub(raw, use_ner=not args.no_ner)
    print(clean)
    if args.show_map:
        for k, v in mapping.items():
            print(f"  {k} -> {v}")