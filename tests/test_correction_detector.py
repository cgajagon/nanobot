"""Tests for correction_detector: clean_phrase, extract_preference_corrections, extract_fact_corrections."""

from __future__ import annotations

import pytest

from nanobot.memory.write.correction_detector import (
    clean_phrase,
    extract_fact_corrections,
    extract_preference_corrections,
)


class TestCleanPhrase:
    """Verify whitespace, punctuation, and article stripping."""

    def test_basic_strip(self) -> None:
        assert clean_phrase("  hello  ") == "hello"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("hello!", "hello"),
            ('"quoted"', "quoted"),
            ("(parens)", "parens"),
            ("[brackets]", "brackets"),
            ("{braces}", "braces"),
        ],
        ids=["exclamation", "quotes", "parens", "brackets", "braces"],
    )
    def test_punctuation_strip(self, raw: str, expected: str) -> None:
        assert clean_phrase(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("the project", "project"),
            ("a task", "task"),
            ("An item", "item"),
            ("THE THING", "THING"),
        ],
        ids=["the", "a", "An", "THE"],
    )
    def test_leading_article_removal(self, raw: str, expected: str) -> None:
        assert clean_phrase(raw) == expected

    def test_multiple_spaces_collapsed(self) -> None:
        assert clean_phrase("too   many   spaces") == "too many spaces"

    def test_empty_string(self) -> None:
        assert clean_phrase("") == ""

    def test_already_clean(self) -> None:
        assert clean_phrase("clean text") == "clean text"


class TestExtractPreferenceCorrections:
    """Verify preference correction pattern extraction."""

    def test_basic_prefer_not(self) -> None:
        result = extract_preference_corrections("I prefer dark mode not light mode")
        assert len(result) == 1
        assert result[0] == ("dark mode", "light mode")

    def test_not_x_but_prefer_y(self) -> None:
        result = extract_preference_corrections("not tabs, but I prefer spaces")
        assert len(result) == 1
        assert result[0][0] == "spaces"
        assert result[0][1] == "tabs"

    def test_with_correction_prefix(self) -> None:
        result = extract_preference_corrections("correction: I prefer vim not emacs")
        assert len(result) == 1
        assert result[0] == ("vim", "emacs")

    def test_now_prefer(self) -> None:
        result = extract_preference_corrections("I now prefer Python not Java")
        assert len(result) == 1
        assert result[0] == ("Python", "Java")

    def test_empty_input(self) -> None:
        assert extract_preference_corrections("") == []

    def test_none_input(self) -> None:
        # The function coerces None via str(content or "")
        assert extract_preference_corrections(None) == []  # type: ignore[arg-type]

    def test_dedup(self) -> None:
        text = "I prefer X not Y. I prefer X not Y."
        result = extract_preference_corrections(text)
        assert len(result) == 1

    def test_same_values_filtered(self) -> None:
        # After cleaning, if new == old, the pair is dropped
        result = extract_preference_corrections("I prefer the thing not thing")
        assert len(result) == 0

    def test_multiple_matches(self) -> None:
        text = "I prefer dark mode not light mode. I prefer tabs not spaces."
        result = extract_preference_corrections(text)
        assert len(result) == 2

    def test_adversarial_long_string(self) -> None:
        long_val = "x" * 5000
        result = extract_preference_corrections(long_val)
        assert isinstance(result, list)

    def test_special_chars_in_values(self) -> None:
        # Should not crash; extraction may or may not match
        result = extract_preference_corrections("I prefer c++ not c#")
        assert isinstance(result, list)


class TestExtractFactCorrections:
    """Verify fact correction pattern extraction."""

    def test_x_is_y_not_z(self) -> None:
        result = extract_fact_corrections("the color is blue not red")
        assert len(result) == 1
        new_fact, old_fact = result[0]
        assert "blue" in new_fact
        assert "red" in old_fact

    def test_x_is_not_y_it_is_z(self) -> None:
        # "X is not Y, it is Z" — group(2)=old, group(3)=new
        result = extract_fact_corrections("the port is not 8080, it is 3000")
        assert len(result) >= 1
        # At least one pair mentions both values with subject embedded
        facts = {v for pair in result for v in pair}
        assert any("8080" in f for f in facts)
        assert any("3000" in f for f in facts)

    def test_with_actually_prefix(self) -> None:
        result = extract_fact_corrections("actually the server is prod not staging")
        assert len(result) == 1
        new_fact, old_fact = result[0]
        assert "prod" in new_fact
        assert "staging" in old_fact

    def test_filters_preference_subjects(self) -> None:
        # Subjects containing "prefer", "want", "use" are filtered
        result = extract_fact_corrections("I prefer is dark not light")
        assert len(result) == 0

    def test_empty_input(self) -> None:
        assert extract_fact_corrections("") == []

    def test_none_input(self) -> None:
        assert extract_fact_corrections(None) == []  # type: ignore[arg-type]

    def test_dedup(self) -> None:
        text = "the color is blue not red. the color is blue not red."
        result = extract_fact_corrections(text)
        assert len(result) == 1

    def test_subject_embedded_in_facts(self) -> None:
        result = extract_fact_corrections("the database is postgres not mysql")
        assert len(result) == 1
        new_fact, old_fact = result[0]
        assert new_fact.startswith("database is")
        assert old_fact.startswith("database is")
