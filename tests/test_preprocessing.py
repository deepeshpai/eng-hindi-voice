"""
Unit tests for pipeline pre-processing helpers — no model loading required.

Tests cover:
- expand_abbreviations()  : ETA/ASAP/FYI → full phrases
- expand_time_expressions(): "5 PM" → "5 in the evening"
- expand_ordinals()       : "1st" → "first", "22nd" → "twenty-second"
- normalize_dates()       : "March 5th" → "5 March"
- expand_currency()       : "$100" → "100 dollars"
- preprocess()            : combined pipeline
- placeholder()           : sentinel name generation
- protect() / restore()   : NER span replacement round-trip (spaCy optional)
- register_custom_entities / protect: custom entity shielding
- apply_tone()            : आप/तुम pronoun substitution
"""
from __future__ import annotations

import pytest

from pipeline.preprocessing import (
    expand_abbreviations,
    expand_currency,
    expand_ordinals,
    expand_time_expressions,
    normalize_dates,
    preprocess,
)
from pipeline.entity_guard import placeholder, protect, restore, register_custom_entities
from pipeline.translator import apply_tone


# ── expand_abbreviations ──────────────────────────────────────────────────────

class TestExpandAbbreviations:
    @pytest.mark.parametrize("abbr,expansion", [
        ("ETA",  "estimated time of arrival"),
        ("ASAP", "as soon as possible"),
        ("FYI",  "for your information"),
        ("WFH",  "work from home"),
        ("OOO",  "out of office"),
        ("TBD",  "to be decided"),
        ("EOD",  "end of day"),
        ("MVP",  "minimum viable product"),
        ("SLA",  "service level agreement"),
        ("POC",  "proof of concept"),
    ])
    def test_known_abbreviations_expanded(self, abbr: str, expansion: str):
        assert expansion in expand_abbreviations(f"The {abbr} is tomorrow.")

    def test_expansion_is_case_sensitive_input(self):
        result = expand_abbreviations("the eta is unknown")
        assert "estimated" not in result

    def test_whole_word_only(self):
        result = expand_abbreviations("DETAIL")
        assert "estimated" not in result

    def test_sentence_with_multiple_abbreviations(self):
        result = expand_abbreviations("ETA ASAP, FYI this is WFH.")
        assert "estimated time of arrival" in result
        assert "as soon as possible" in result
        assert "for your information" in result
        assert "work from home" in result

    def test_non_abbreviation_unchanged(self):
        text = "Can we schedule a meeting tomorrow?"
        assert expand_abbreviations(text) == text

    def test_empty_string(self):
        assert expand_abbreviations("") == ""

    def test_abbreviation_at_start_of_sentence(self):
        result = expand_abbreviations("ASAP please send the report.")
        assert result.startswith("as soon as possible")

    def test_abbreviation_at_end_of_sentence(self):
        result = expand_abbreviations("Please do this ASAP")
        assert result.endswith("as soon as possible")


# ── expand_time_expressions ───────────────────────────────────────────────────

class TestExpandTimeExpressions:
    @pytest.mark.parametrize("text,expected_fragment", [
        ("Meeting at 5 PM.",   "5 in the evening"),
        ("Call at 10 AM.",     "10 in the morning"),
        ("Available at 12 PM", "12 in the evening"),
        ("Wake up at 6 AM.",   "6 in the morning"),
        ("Standup at 9 am.",   "9 in the morning"),
        ("Call at 3 pm.",      "3 in the evening"),
    ])
    def test_pm_and_am_replaced(self, text: str, expected_fragment: str):
        result = expand_time_expressions(text)
        assert expected_fragment in result
        assert "PM" not in result.upper().split()
        assert "AM" not in result.upper().split()

    def test_no_time_expression_unchanged(self):
        text = "The meeting is tomorrow."
        assert expand_time_expressions(text) == text

    def test_multiple_times_in_sentence(self):
        result = expand_time_expressions("From 9 AM to 5 PM.")
        assert "9 in the morning" in result
        assert "5 in the evening" in result

    def test_two_digit_hour(self):
        result = expand_time_expressions("Meeting at 10 PM.")
        assert "10 in the evening" in result

    def test_empty_string(self):
        assert expand_time_expressions("") == ""


# ── placeholder ───────────────────────────────────────────────────────────────

class TestPlaceholder:
    def test_known_index_words(self):
        assert placeholder(0) == "Nentityzero"
        assert placeholder(1) == "Nentityone"
        assert placeholder(5) == "Nentityfive"
        assert placeholder(20) == "Nentitytwenty"

    def test_all_alphabetic(self):
        for i in range(21):
            ph = placeholder(i)
            assert ph.isalpha(), f"Placeholder {ph!r} for index {i} is not all-alphabetic"

    def test_fallback_beyond_word_list(self):
        ph = placeholder(25)
        assert ph == "Nentity25"

    def test_prefix_is_present(self):
        for i in range(5):
            assert placeholder(i).startswith("Nentity")

    def test_unique_per_index(self):
        placeholders = [placeholder(i) for i in range(15)]
        assert len(placeholders) == len(set(placeholders))


# ── protect / restore ─────────────────────────────────────────────────────────

class TestEntityProtection:
    """
    These tests work whether or not spaCy is installed.
    When spaCy is unavailable, protect() returns the text unchanged with
    an empty mapping — the restore tests still validate that contract.
    """

    def test_round_trip_no_entities(self):
        text = "The meeting is tomorrow at noon."
        protected, mapping = protect(text)
        restored = restore(protected, mapping)
        assert restored == protected

    def test_empty_string(self):
        protected, mapping = protect("")
        assert protected == ""
        assert mapping == {}

    def test_restore_empty_mapping(self):
        assert restore("hello world", {}) == "hello world"

    def test_restore_with_explicit_mapping(self):
        mapping = {"Nentityzero": "Google Meet"}
        text_with_placeholder = "Can we move Nentityzero to tomorrow?"
        restored = restore(text_with_placeholder, mapping)
        assert "Google Meet" in restored
        assert "Nentityzero" not in restored

    def test_restore_is_case_tolerant(self):
        """MT model may alter the placeholder's case — restore must still work."""
        mapping = {"Nentityzero": "Asterisk"}
        mangled = "Can we use nentityzero for the SIP trunk?"
        restored = restore(mangled, mapping)
        assert "Asterisk" in restored

    def test_restore_longest_first_avoids_collision(self):
        mapping = {
            "Nentityone":    "Google",
            "Nentityonetwo": "Google Meet",
        }
        text = "Join Nentityonetwo using Nentityone account."
        restored = restore(text, mapping)
        assert "Google Meet" in restored
        assert "Google" in restored

    @pytest.mark.parametrize("text", [
        "Can we move the Google Meet to 5 PM tomorrow?",
        "Please share the Asterisk SIP trunk config.",
        "John from Accenture called about the project.",
    ])
    def test_round_trip_preserves_entities(self, text: str):
        protected, mapping = protect(text)
        restored = restore(protected, mapping)
        for original in mapping.values():
            assert original in restored


# ── expand_ordinals ───────────────────────────────────────────────────────────

class TestExpandOrdinals:
    @pytest.mark.parametrize("ordinal,word", [
        ("1st", "first"),
        ("2nd", "second"),
        ("3rd", "third"),
        ("5th", "fifth"),
        ("12th", "twelfth"),
        ("21st", "twenty-first"),
        ("22nd", "twenty-second"),
        ("23rd", "twenty-third"),
        ("30th", "thirtieth"),
        ("31st", "thirty-first"),
    ])
    def test_known_ordinals_expanded(self, ordinal: str, word: str):
        result = expand_ordinals(f"He finished {ordinal} in the race.")
        assert word in result
        assert ordinal.lower() not in result.lower()

    def test_case_insensitive_input(self):
        result = expand_ordinals("She won 1ST place.")
        assert "first" in result.lower()

    def test_preserves_non_ordinal_numbers(self):
        text = "The price is 100 dollars."
        assert expand_ordinals(text) == text

    def test_empty_string(self):
        assert expand_ordinals("") == ""


# ── normalize_dates ───────────────────────────────────────────────────────────

class TestNormalizeDates:
    @pytest.mark.parametrize("text,expected_day,expected_month", [
        ("March 5th",     "5",  "March"),
        ("January 1st",   "1",  "January"),
        ("December 31st", "31", "December"),
        ("Feb 14",        "14", "February"),
        ("Jul 4th",       "4",  "July"),
        ("oct 20",        "20", "October"),
    ])
    def test_month_first_reordered(self, text: str, expected_day: str, expected_month: str):
        result = normalize_dates(text)
        assert result == f"{expected_day} {expected_month}"

    def test_date_in_sentence(self):
        result = normalize_dates("The meeting is on March 5th at noon.")
        assert "5 March" in result
        # Ordinal suffix stripped
        assert "5th" not in result

    def test_may_not_confused_with_modal_verb(self):
        """'may I' should NOT be altered — 'may' must be followed by a digit."""
        text = "May I ask a question?"
        result = normalize_dates(text)
        assert result == text

    def test_empty_string(self):
        assert normalize_dates("") == ""

    def test_month_only_unchanged(self):
        """A bare month name with no following digit should not be altered."""
        text = "See you in March."
        assert normalize_dates(text) == text


# ── expand_currency ───────────────────────────────────────────────────────────

class TestExpandCurrency:
    @pytest.mark.parametrize("text,expected", [
        ("$100",      "100 dollars"),
        ("€50",       "50 euros"),
        ("£200",      "200 pounds"),
        ("₹500",      "500 rupees"),
        ("¥1000",     "1000 yen"),
        ("$1,000",    "1000 dollars"),
        ("$99.99",    "99.99 dollars"),
    ])
    def test_currency_symbols_expanded(self, text: str, expected: str):
        assert expand_currency(text) == expected

    def test_currency_in_sentence(self):
        result = expand_currency("The project costs $50,000 this quarter.")
        assert "50000 dollars" in result

    def test_no_currency_unchanged(self):
        text = "The meeting is tomorrow."
        assert expand_currency(text) == text

    def test_empty_string(self):
        assert expand_currency("") == ""


# ── preprocess (combined) ─────────────────────────────────────────────────────

class TestPreprocess:
    def test_full_pipeline_example(self):
        # "January 1st" → normalize_dates strips the suffix → "1 January" (bare digit)
        # A standalone ordinal like "5th" in a non-date context → "fifth"
        result = preprocess(
            "Can we move to 5 PM tomorrow, ETA by January 1st? This is our 5th attempt."
        )
        assert "in the evening" in result           # time expansion
        assert "PM" not in result                   # time normalised
        assert "estimated time of arrival" in result  # abbreviation
        assert "1 January" in result                # date normalised (ordinal suffix stripped)
        assert "fifth" in result                    # standalone ordinal expanded

    def test_currency_and_date_combined(self):
        result = preprocess("Budget is $5,000 due by March 31st.")
        assert "5000 dollars" in result
        assert "31 March" in result

    def test_empty_string(self):
        assert preprocess("") == ""


# ── custom entity protection ──────────────────────────────────────────────────

class TestCustomEntityProtection:
    def test_custom_entity_is_protected(self):
        """A registered custom entity must survive protect/restore unchanged."""
        register_custom_entities(["Asterisk"])
        protected, mapping = protect("We use Asterisk for SIP.")
        assert "Asterisk" not in protected        # replaced by placeholder
        restored = restore(protected, mapping)
        assert "Asterisk" in restored             # restored intact

    def test_custom_entity_case_insensitive(self):
        register_custom_entities(["Twilio"])
        protected, mapping = protect("twilio handles the calls")
        restored = restore(protected, mapping)
        assert "twilio" in restored.lower()

    def test_multi_word_custom_entity(self):
        register_custom_entities(["Google Meet"])
        protected, mapping = protect("Join Google Meet now.")
        assert "Google Meet" not in protected
        restored = restore(protected, mapping)
        assert "Google Meet" in restored

    def test_register_duplicate_ignored(self):
        """Calling register_custom_entities twice with the same entity is safe."""
        from pipeline.entity_guard import _CUSTOM_ENTITIES
        before = len(_CUSTOM_ENTITIES)
        register_custom_entities(["UniqueTestEntity123"])
        after_first = len(_CUSTOM_ENTITIES)
        register_custom_entities(["UniqueTestEntity123"])  # duplicate
        assert len(_CUSTOM_ENTITIES) == after_first       # no growth


# ── apply_tone ────────────────────────────────────────────────────────────────

class TestApplyTone:
    def test_formal_replaces_tum_with_aap(self):
        hindi = "तुम कब आओगे?"
        result = apply_tone(hindi, "formal")
        assert "आप" in result
        assert "तुम" not in result

    def test_casual_replaces_aap_with_casual_form(self):
        hindi = "आप कब आएंगे?"
        result = apply_tone(hindi, "casual")
        assert any(form in result for form in ("तू", "तुम")), f"No casual pronoun found: {result!r}"
        assert "आप" not in result

    def test_auto_leaves_text_unchanged(self):
        hindi = "वे कल आएंगे।"
        assert apply_tone(hindi, "auto") == hindi

    def test_unknown_tone_leaves_text_unchanged(self):
        hindi = "नमस्ते।"
        assert apply_tone(hindi, "neutral") == hindi

    def test_formal_possessive_replaced(self):
        hindi = "तुम्हारा काम अच्छा है।"
        result = apply_tone(hindi, "formal")
        assert "आपका" in result

    def test_casual_possessive_replaced(self):
        hindi = "आपका नाम क्या है?"
        result = apply_tone(hindi, "casual")
        assert "तुम्हारा" in result

    def test_empty_string(self):
        assert apply_tone("", "formal") == ""
        assert apply_tone("", "casual") == ""
        assert apply_tone("", "auto") == ""
