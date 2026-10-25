import math

from cs2model.pipeline import ml_support


def test_favourite_supports_over_and_underdog_supports_under():
    assert ml_support("over", 0.68) == "supports"
    assert ml_support("under", 0.68) == "against"
    assert ml_support("under", 0.30) == "supports"
    assert ml_support("over", 0.30) == "against"


def test_coin_flip_is_neutral():
    assert ml_support("over", 0.52) == "neutral"
    assert ml_support("under", 0.46) == "neutral"


def test_missing_odds_or_side_is_blank():
    assert ml_support("over", math.nan) == ""
    assert ml_support("over", None) == ""
    assert ml_support("", 0.7) == ""
