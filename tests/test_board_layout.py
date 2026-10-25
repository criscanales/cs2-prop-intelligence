import re
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "cs2model" / "dashboard" / "app.py").read_text()
BOARD = SRC[SRC.index('<div class="sect">Full board</div>'):SRC.index('<div class="sect">Actions</div>')]


def test_board_column_order_puts_key_fields_first():
    cols = eval(re.search(r"cols = (\[[^\]]*\])", BOARD).group(1))
    assert cols[:8] == ["logo", "player", "matchup", "prop", "line", "side", "edge", "ml"]
    assert {"ev", "tier", "model_mean", "p_over", "p_under", "start", "n_maps", "source", "why"} <= set(cols)
    assert "chg" not in cols and "ml_fit" not in cols


def test_board_has_no_against_labels_or_chg_column():
    assert "against" not in BOARD.replace('"against"', "")
    assert '"chg"' not in BOARD


def test_edge_stays_bright_and_ev_is_muted():
    assert 'c == "edge"' in BOARD and "#86efac" in BOARD
    assert 'c == "ev"' in BOARD and "#7fa68e" in BOARD


def test_rows_are_neutral_alternating_charcoal_with_padding():
    assert "rgba(255,122,26" not in BOARD
    assert "#151b23" in BOARD and "#11161d" in BOARD and "% 2" in BOARD
    assert "row_height=38" in BOARD


def test_team_ml_only_shown_with_verified_prematch_odds():
    assert "prematch_odds_ok(" in BOARD and "_pw.where(_ok)" in BOARD


def test_stale_selection_after_filtering_is_ignored():
    """A row picked before a filter shrinks the board must not index past the new table."""
    assert "k = t.index[rows[0]] if rows and rows[0] < len(t) else None" in BOARD + SRC


def test_missing_logo_renders_blank_not_none():
    assert 'LOGOS.get(int(t)) or ""' in BOARD


def test_team_ml_shows_percentage_only():
    assert 'ml=[f"{p:.0%}" if p == p else "" for p in _pw]' in BOARD
    assert "american(p)" not in BOARD
