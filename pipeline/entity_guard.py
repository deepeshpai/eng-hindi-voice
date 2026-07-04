"""
Named-entity protection for neural machine translation.

Why this exists
---------------
MT models translate proper nouns literally:
  "Asterisk" (telephony framework) → "तारांकन" (punctuation symbol)
  "Google Meet" → "गूगल मिलना"

We guard against this with a placeholder round-trip:
  1. Pre-translate : detect entity spans; replace with sentinel tokens
  2. Translate     : MT model sees opaque tokens, copies them through
  3. Post-translate: restore original spans in the output

Two detection mechanisms (applied in priority order)
----------------------------------------------------
1. Custom entity registry  — exact-match, case-insensitive, whole-word.
   Populated at startup from ASRConfig.hotwords so technical terms like
   "Asterisk", "Twilio", "Google Meet", "SIP", "WebRTC" are ALWAYS protected
   regardless of whether spaCy detects them.

2. spaCy NER  — detects PERSON, ORG, GPE, PRODUCT, etc. spans.  Only applied
   to regions not already covered by a custom entity.

Placeholder design
------------------
Sentinels are purely alphabetic ("Nentityzero", "Nentityone") — NOT
digit-bearing ("VBOT0").  NMT models transliterate digit tokens into
Devanagari, breaking the literal restore.  All-letter tokens survive
the round-trip reliably.

spaCy availability
------------------
If spaCy or en_core_web_sm is not installed, protection gracefully falls back
to the custom entity registry only.
"""
from __future__ import annotations

import re
from typing import Optional

# ── Sentinel vocabulary ───────────────────────────────────────────────────────

_PLACEHOLDER_PREFIX = "Nentity"
_INDEX_WORDS = [
    "zero", "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
    "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
]


def placeholder(idx: int) -> str:
    suffix = _INDEX_WORDS[idx] if idx < len(_INDEX_WORDS) else str(idx)
    return f"{_PLACEHOLDER_PREFIX}{suffix}"


# ── Custom entity registry ────────────────────────────────────────────────────

_CUSTOM_ENTITIES: list[str] = []


def register_custom_entities(entities: list[str]) -> None:
    """
    Register domain-specific entities that are ALWAYS protected from translation.

    Unlike spaCy NER, custom entities use exact string matching (case-insensitive,
    whole-word boundaries).  Useful for technical terms that spaCy may tag as
    common nouns or miss entirely: "Asterisk", "Twilio", "SIP", "WebRTC".

    Call once at startup (e.g. from ModelRegistry.load()) with the hotwords list.
    Safe to call multiple times — duplicates are ignored.
    """
    for raw in entities:
        entity = raw.strip()
        if entity and entity not in _CUSTOM_ENTITIES:
            _CUSTOM_ENTITIES.append(entity)


def _custom_entity_spans(text: str) -> list[tuple[int, int]]:
    """Find char-offset spans of registered custom entities (whole-word, case-insensitive)."""
    spans: list[tuple[int, int]] = []
    # Longer entities first to prevent partial matches within longer names
    for entity in sorted(_CUSTOM_ENTITIES, key=len, reverse=True):
        pattern = re.compile(r"\b" + re.escape(entity) + r"\b", re.IGNORECASE)
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end()))
    return spans


# ── spaCy NER (lazy-loaded) ───────────────────────────────────────────────────

_NER_PROTECT_LABELS: frozenset[str] = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "FAC",
    "PRODUCT", "NORP", "WORK_OF_ART", "EVENT",
})

_spacy_nlp: Optional[object] = None
_spacy_unavailable: bool = False


def _get_ner() -> Optional[object]:
    global _spacy_nlp, _spacy_unavailable
    if _spacy_nlp is not None:
        return _spacy_nlp
    if _spacy_unavailable:
        return None
    try:
        import spacy  # type: ignore
        _spacy_nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "textcat"])
    except Exception:
        _spacy_unavailable = True
        return None
    return _spacy_nlp


# ── Span trimming ─────────────────────────────────────────────────────────────
# spaCy sometimes over-extends entity spans:
#   "the Google Meet to 5 PM" tagged as EVENT
# We trim to the maximal consecutive PROPN/NOUN run.

_LEAD_TRIM_POS: frozenset[str] = frozenset({"DET", "ADP", "PART", "CCONJ", "SCONJ", "PUNCT"})
_PROPN_POS: frozenset[str] = frozenset({"PROPN", "NOUN", "X"})
_PROPN_GLUE: frozenset[str] = frozenset({"-", "'", "\u2019", ".", "&", "+"})


def _trim_to_propn_core(doc, ent) -> Optional[tuple[int, int]]:
    """
    Return (start_char, end_char) of the proper-noun core of *ent*, or None.

    Algorithm:
      1. Skip leading DET/ADP/stopwords.
      2. Extend forward through consecutive PROPN/NOUN/X tokens only.
      Stops at the first preposition, numeral, etc.
    """
    start = ent.start
    end = ent.end  # exclusive

    while start < end:
        tok = doc[start]
        if tok.pos_ in _LEAD_TRIM_POS or tok.is_stop:
            start += 1
        else:
            break

    if start >= end:
        return None

    propn_end = start
    while propn_end < end:
        tok = doc[propn_end]
        if tok.pos_ in _PROPN_POS or tok.text in _PROPN_GLUE:
            propn_end += 1
        else:
            break

    if propn_end <= start:
        return None
    return doc[start].idx, doc[propn_end - 1].idx + len(doc[propn_end - 1].text)


def _ner_spans(text: str) -> list[tuple[int, int]]:
    """Return char-offset spans of spaCy NER entities, longest first."""
    nlp = _get_ner()
    if nlp is None:
        return []
    doc = nlp(text)  # type: ignore[operator]
    spans: list[tuple[int, int]] = []
    for ent in doc.ents:  # type: ignore[union-attr]
        if ent.label_ not in _NER_PROTECT_LABELS or not ent.text.strip():
            continue
        trimmed = _trim_to_propn_core(doc, ent)
        if trimmed:
            spans.append(trimmed)
    spans.sort(key=lambda s: s[1] - s[0], reverse=True)
    return spans


# ── Span merging ──────────────────────────────────────────────────────────────

def _merge_spans(
    custom: list[tuple[int, int]],
    ner: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """
    Combine custom-entity and NER spans into a non-overlapping set.

    Custom entities have priority: any NER span that overlaps with a custom
    entity span is discarded.  The result is sorted by start position.
    """
    selected = list(custom)
    for start, end in ner:
        overlaps = any(s < end and e > start for s, e in selected)
        if not overlaps:
            selected.append((start, end))
    selected.sort(key=lambda s: s[0])
    return selected


# ── Public API ────────────────────────────────────────────────────────────────

def protect(text: str) -> tuple[str, dict[str, str]]:
    """
    Replace entity spans with opaque placeholders.

    Custom entities are detected first (higher priority) and always protected.
    spaCy NER covers spans not already matched by custom entities.

    Returns
    -------
    (protected_text, mapping)
        mapping: {placeholder_token: original_span}
    """
    mapping: dict[str, str] = {}
    idx = 0

    merged = _merge_spans(_custom_entity_spans(text), _ner_spans(text))

    # Replace in reverse order so earlier char offsets remain valid
    for start, end in sorted(merged, key=lambda s: s[0], reverse=True):
        span_text = text[start:end]
        key = placeholder(idx)
        mapping[key] = span_text
        text = text[:start] + key + text[end:]
        idx += 1

    return text, mapping


def _restore_pattern(ph: str) -> re.Pattern:
    """Tolerant pattern: matches placeholder even if MT inserted spaces or changed case."""
    suffix = ph[len(_PLACEHOLDER_PREFIX):]
    return re.compile(
        re.escape(_PLACEHOLDER_PREFIX) + r"[\s\u200c\u200d]*" + re.escape(suffix),
        re.IGNORECASE,
    )


def restore(text: str, mapping: dict[str, str]) -> str:
    """Replace placeholders in translated text with their original spans."""
    for ph in sorted(mapping, key=len, reverse=True):  # longest first
        text = _restore_pattern(ph).sub(
            lambda _m, orig=mapping[ph]: orig, text
        )
    return text
