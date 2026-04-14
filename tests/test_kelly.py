"""Tests for Kelly criterion and position sizing."""

from tennis_edge.strategy.kelly import kelly_fraction, fractional_kelly, edge, expected_value
from tennis_edge.strategy.sizing import PositionSizer


def test_kelly_no_edge():
    """When model agrees with market, Kelly = 0."""
    assert kelly_fraction(0.5, 0.5) == 0.0
    assert kelly_fraction(0.7, 0.7) == 0.0


def test_kelly_positive_edge():
    """When model thinks YES is more likely than market."""
    f = kelly_fraction(0.6, 0.5)
    assert f > 0  # Should bet YES
    assert abs(f - 0.2) < 0.01  # (0.6 - 0.5) / (1 - 0.5) = 0.2


def test_kelly_negative_edge():
    """When model thinks NO is more likely than market."""
    f = kelly_fraction(0.4, 0.6)
    assert f < 0  # Should bet NO


def test_fractional_kelly():
    f = fractional_kelly(0.6, 0.5, fraction=0.25)
    assert abs(f - 0.05) < 0.01  # 0.25 * 0.2 = 0.05


def test_edge_calculation():
    assert abs(edge(0.7, 0.6) - 0.1) < 0.001
    assert abs(edge(0.3, 0.6) - (-0.3)) < 0.001


def test_expected_value():
    ev = expected_value(0.6, 0.5)
    assert ev > 0  # Positive EV
    assert abs(ev - 0.2) < 0.01  # 0.6/0.5 - 1 = 0.2


def test_sizer_below_min_edge():
    sizer = PositionSizer(bankroll=1000, min_edge=0.05)
    result = sizer.size(model_prob=0.52, market_price_cents=50)
    assert result is None  # 2% edge < 5% min


def test_sizer_above_min_edge():
    sizer = PositionSizer(bankroll=1000, min_edge=0.03, kelly_fraction=0.25)
    result = sizer.size(model_prob=0.65, market_price_cents=50)
    assert result is not None
    assert result.side == "yes"
    assert result.num_contracts >= 1
    assert result.bet_amount > 0


def test_sizer_no_side():
    sizer = PositionSizer(bankroll=1000, min_edge=0.03)
    result = sizer.size(model_prob=0.35, market_price_cents=50)
    assert result is not None
    assert result.side == "no"
