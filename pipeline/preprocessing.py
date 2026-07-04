"""
Text pre-processing for the translation pipeline.

Responsibilities
----------------
1. Abbreviation expansion      ETA → "estimated time of arrival"
2. Time-expression expansion   5 PM → "5 in the evening"
3. Ordinal expansion           "1st" → "first", "22nd" → "twenty-second"
4. Date normalisation          "March 5th" → "5 March"  (day-first matches
                               Hindi natural word order and MT training data)
5. Currency expansion          "$100" → "100 dollars", "₹500" → "500 rupees"

All functions are pure (str → str) with no model loading.

Ordering in preprocess()
------------------------
  expand_currency   — before dates so "£5 on March 5th" is handled cleanly
  normalize_dates   — strips ordinal suffix from day number inside a date
  expand_abbreviations
  expand_time_expressions
  expand_ordinals   — handles remaining standalone ordinals after dates
"""
from __future__ import annotations

import re

# ── Abbreviation table ────────────────────────────────────────────────────────

_ABBREVIATIONS: dict[str, str] = {
    "ETA":  "estimated time of arrival",
    "ASAP": "as soon as possible",
    "FYI":  "for your information",
    "WFH":  "work from home",
    "OOO":  "out of office",
    "TBD":  "to be decided",
    "TBH":  "to be honest",
    "IMO":  "in my opinion",
    "IMHO": "in my humble opinion",
    "LGTM": "looks good to me",
    "EOD":  "end of day",
    "COB":  "close of business",
    "OKR":  "objective and key result",
    "KPI":  "key performance indicator",
    "MOU":  "memorandum of understanding",
    "POC":  "proof of concept",
    "MVP":  "minimum viable product",
    "SLA":  "service level agreement",
}

# Whole-word, longest-first so "IMHO" matches before "IMO"
_ABBR_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\b"
)


def expand_abbreviations(text: str) -> str:
    """Replace known ALL-CAPS abbreviations with their full English phrase."""
    return _ABBR_RE.sub(lambda m: _ABBREVIATIONS[m.group(1).upper()], text)


# ── Time-expression expansion ─────────────────────────────────────────────────
# opus-mt and smaller models often translate "PM" as "Prime Minister".
# Expanding "5 PM" → "5 in the evening" feeds the model a phrase it
# learned from its parallel training corpus.

_TIME_RE = re.compile(r"\b(\d{1,2})\s*(AM|PM)\b", re.IGNORECASE)


def expand_time_expressions(text: str) -> str:
    """Replace '5 PM' / '9 AM' with natural-language equivalents."""
    def _repl(m: re.Match) -> str:
        hour   = m.group(1)
        period = m.group(2).upper()
        suffix = "in the morning" if period == "AM" else "in the evening"
        return f"{hour} {suffix}"
    return _TIME_RE.sub(_repl, text)


# ── Ordinal number expansion ──────────────────────────────────────────────────
# Hindi MT models often struggle with ordinal suffixes ("1st", "22nd").
# Expanding them to words before translation avoids mistranslation.
#
# Note: normalize_dates() already strips ordinal suffixes from day numbers
# inside date phrases ("March 5th" → "5 March"), so expand_ordinals() only
# sees standalone ordinals like "the 5th floor" or "his 1st attempt".

_ORDINALS: dict[str, str] = {
    "1st":  "first",          "2nd":  "second",       "3rd":  "third",
    "4th":  "fourth",         "5th":  "fifth",         "6th":  "sixth",
    "7th":  "seventh",        "8th":  "eighth",        "9th":  "ninth",
    "10th": "tenth",          "11th": "eleventh",      "12th": "twelfth",
    "13th": "thirteenth",     "14th": "fourteenth",    "15th": "fifteenth",
    "16th": "sixteenth",      "17th": "seventeenth",   "18th": "eighteenth",
    "19th": "nineteenth",     "20th": "twentieth",
    "21st": "twenty-first",   "22nd": "twenty-second", "23rd": "twenty-third",
    "24th": "twenty-fourth",  "25th": "twenty-fifth",  "26th": "twenty-sixth",
    "27th": "twenty-seventh", "28th": "twenty-eighth", "29th": "twenty-ninth",
    "30th": "thirtieth",      "31st": "thirty-first",
}

_ORDINAL_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_ORDINALS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand_ordinals(text: str) -> str:
    """Expand ordinal numerals to words: '1st' → 'first', '22nd' → 'twenty-second'."""
    return _ORDINAL_RE.sub(lambda m: _ORDINALS[m.group(1).lower()], text)


# ── Date normalisation ────────────────────────────────────────────────────────
# English uses month-first order ("March 5th") while Hindi uses day-first
# ("5 मार्च").  Reordering before translation aligns with the MT model's
# training distribution and significantly improves date localisation.
#
# Pattern: "March 5th" / "January 1" / "Dec 31st" → "5 March" / "1 January"

_MONTH_LONG_TO_FULL: dict[str, str] = {
    "january": "January", "february": "February", "march": "March",
    "april": "April", "may": "May", "june": "June", "july": "July",
    "august": "August", "september": "September", "october": "October",
    "november": "November", "december": "December",
}
_MONTH_SHORT_TO_FULL: dict[str, str] = {
    "jan": "January", "feb": "February", "mar": "March",
    "apr": "April", "jun": "June", "jul": "July",
    "aug": "August", "sep": "September", "sept": "September",
    "oct": "October", "nov": "November", "dec": "December",
}
_MONTH_CANON: dict[str, str] = {**_MONTH_LONG_TO_FULL, **_MONTH_SHORT_TO_FULL}

_MONTH_PAT = "|".join(
    re.escape(m) for m in sorted(_MONTH_CANON, key=len, reverse=True)
)

# Matches "March 5th", "Jan 1", "december 31st" etc.
# Captures: group(1) = month name, group(2) = day number (with optional ordinal suffix)
_DATE_MONTH_FIRST_RE = re.compile(
    rf"\b({_MONTH_PAT})\s+(\d{{1,2}}(?:st|nd|rd|th)?)\b",
    re.IGNORECASE,
)


def normalize_dates(text: str) -> str:
    """
    Reorder month-first dates to day-first for better Hindi MT output.
    "March 5th" → "5 March"  |  "Jan 1" → "1 January"
    """
    def _repl(m: re.Match) -> str:
        month = _MONTH_CANON[m.group(1).lower()]
        # Strip ordinal suffix from the day number
        day = re.sub(r"(st|nd|rd|th)$", "", m.group(2), flags=re.IGNORECASE)
        return f"{day} {month}"
    return _DATE_MONTH_FIRST_RE.sub(_repl, text)


# ── Currency expansion ────────────────────────────────────────────────────────
# Currency symbols are opaque to MT models trained on text.  Expanding them to
# natural-language phrases ("$100" → "100 dollars") avoids bizarre output.

_CURRENCY_UNITS: dict[str, str] = {
    "$": "dollars",
    "€": "euros",
    "£": "pounds",
    "₹": "rupees",
    "¥": "yen",
}

_CURRENCY_RE = re.compile(
    r"([€£₹¥$])(\d+(?:,\d{3})*(?:\.\d+)?)"
)


def expand_currency(text: str) -> str:
    """
    Expand currency symbols to words.
    "$100" → "100 dollars"  |  "₹1,000" → "1000 rupees"
    """
    def _repl(m: re.Match) -> str:
        symbol = m.group(1)
        # Remove thousands separators before handing to MT
        amount = m.group(2).replace(",", "")
        unit = _CURRENCY_UNITS.get(symbol, symbol)
        return f"{amount} {unit}"
    return _CURRENCY_RE.sub(_repl, text)


# ── Combined preprocessor ─────────────────────────────────────────────────────

def preprocess(text: str) -> str:
    """Apply all pre-processing steps in the correct order."""
    text = expand_currency(text)          # "$100" → "100 dollars"
    text = normalize_dates(text)          # "March 5th" → "5 March"
    text = expand_abbreviations(text)     # "ETA" → "estimated time of arrival"
    text = expand_time_expressions(text)  # "5 PM" → "5 in the evening"
    text = expand_ordinals(text)          # remaining "1st" → "first"
    return text
