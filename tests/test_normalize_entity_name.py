"""Tests for normalize_entity_name in memory._text."""

from __future__ import annotations

import pytest

from nanobot.memory._text import normalize_entity_name


class TestNormalizeEntityName:
    """normalize_entity_name strips possessives, titles, punctuation, normalizes."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            # Basic lowercasing and whitespace
            ("Carlos", "carlos"),
            ("  Carlos  ", "carlos"),
            ("Carlos Gajardo", "carlos_gajardo"),
            # Possessives (straight and smart quotes)
            ("User's", "user"),
            ("Carlos's", "carlos"),
            ("User\u2019s", "user"),  # smart quote '
            ("User\u2018s", "user"),  # smart quote '
            # Titles stripped at start
            ("Dr. Smith", "smith"),
            ("Mr. Jones", "jones"),
            ("Mrs. Williams", "williams"),
            ("Ms. Davis", "davis"),
            ("Prof. Lee", "lee"),
            # Titles NOT stripped mid-string
            ("Visit Dr. Smith", "visit_dr_smith"),
            # Punctuation stripped (except hyphens/underscores)
            ("O'Brien", "obrien"),
            ("vue-router", "vue-router"),
            ("my_project", "my_project"),
            ("hello.world", "helloworld"),
            # Unicode NFKC
            ("\ufb01nance", "finance"),  # fi ligature
            ("caf\u00e9", "caf\u00e9"),  # accented char preserved after NFKC
            # Empty and whitespace
            ("", ""),
            ("   ", ""),
            # Multiple spaces become single underscore
            ("New   York   City", "new_york_city"),
        ],
    )
    def test_normalization(self, raw: str, expected: str) -> None:
        assert normalize_entity_name(raw) == expected
