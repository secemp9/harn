from __future__ import annotations

from harnify_tui.fuzzy import fuzzyFilter, fuzzyMatch


def test_fuzzy_match_empty_query_matches_everything() -> None:
    result = fuzzyMatch("", "anything")
    assert result.matches is True
    assert result.score == 0


def test_fuzzy_match_exact_and_boundary_matches_score_better() -> None:
    exact = fuzzyMatch("test", "test")
    boundary = fuzzyMatch("fb", "foo-bar")
    not_boundary = fuzzyMatch("fb", "afbx")

    assert exact.matches is True
    assert exact.score < 0
    assert boundary.matches is True
    assert not_boundary.matches is True
    assert boundary.score < not_boundary.score


def test_fuzzy_match_supports_swapped_alpha_numeric_queries() -> None:
    result = fuzzyMatch("codex52", "gpt-5.2-codex")
    assert result.matches is True


def test_fuzzy_filter_sorts_by_match_quality() -> None:
    items = ["a_p_p", "app", "application"]
    result = fuzzyFilter(items, "app", lambda item: item)
    assert result[0] == "app"


def test_fuzzy_filter_handles_multi_token_queries() -> None:
    items = ["packages/tui/src/autocomplete.ts", "packages/ai/src/autocomplete.ts", "README.md"]
    result = fuzzyFilter(items, "tui auto", lambda item: item)
    assert result == ["packages/tui/src/autocomplete.ts"]
