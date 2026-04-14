"""Kelly criterion for optimal bet sizing."""

from __future__ import annotations


def kelly_fraction(model_prob: float, market_prob: float) -> float:
    """Full Kelly criterion for a binary outcome.

    For a bet on YES at market implied probability q (price):
        f* = (p - q) / (1 - q)
    where p = model_prob, q = market_prob.

    If negative, indicates edge is on the NO side:
        f_no* = (q - p) / q

    Returns:
        Positive = bet YES, negative = bet NO, magnitude = fraction of bankroll.
    """
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    if model_prob <= 0 or model_prob >= 1:
        return 0.0

    edge_yes = model_prob - market_prob
    if edge_yes > 0:
        return edge_yes / (1.0 - market_prob)
    else:
        # Edge is on NO side
        edge_no = market_prob - model_prob
        return -(edge_no / market_prob)


def fractional_kelly(
    model_prob: float,
    market_prob: float,
    fraction: float = 0.25,
) -> float:
    """Fractional Kelly: scale down full Kelly by a conservative fraction.

    Returns signed value: positive = YES, negative = NO.
    """
    return kelly_fraction(model_prob, market_prob) * fraction


def expected_value(model_prob: float, market_prob: float) -> float:
    """Expected value per dollar risked on YES side.

    EV = p * (1 - q) / q - (1 - p)
    Simplified: EV = p/q - 1
    """
    if market_prob <= 0:
        return 0.0
    return model_prob / market_prob - 1.0


def edge(model_prob: float, market_prob: float) -> float:
    """Raw edge: difference between model and market probability."""
    return model_prob - market_prob
