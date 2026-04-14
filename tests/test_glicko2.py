"""Tests for Glicko-2 algorithm against known examples."""

from tennis_edge.ratings.glicko2 import Glicko2Engine, Glicko2Rating


def test_glicko2_known_example():
    """Verify against Glickman's example from the paper.

    Player: mu=1500, phi=200, sigma=0.06
    Opponents:
        1: mu=1400, phi=30  -> win
        2: mu=1550, phi=100 -> loss
        3: mu=1700, phi=300 -> loss
    Expected: mu≈1464, phi≈151.52, sigma≈0.05999
    """
    engine = Glicko2Engine(tau=0.5)

    player = Glicko2Rating(mu=1500, phi=200, sigma=0.06)
    opponents = [
        Glicko2Rating(mu=1400, phi=30, sigma=0.06),
        Glicko2Rating(mu=1550, phi=100, sigma=0.06),
        Glicko2Rating(mu=1700, phi=300, sigma=0.06),
    ]
    outcomes = [1.0, 0.0, 0.0]

    result = engine.rate(player, opponents, outcomes)

    assert abs(result.mu - 1464.06) < 1.0, f"mu={result.mu}"
    assert abs(result.phi - 151.52) < 2.0, f"phi={result.phi}"
    assert result.sigma > 0.059 and result.sigma < 0.061, f"sigma={result.sigma}"


def test_no_games_increases_phi():
    """If a player has no games, phi should increase."""
    engine = Glicko2Engine(tau=0.5)
    player = Glicko2Rating(mu=1500, phi=200, sigma=0.06)

    result = engine.rate(player, [], [])

    assert result.mu == 1500
    assert result.phi > 200
    assert result.sigma == 0.06


def test_expected_score():
    """Test expected score calculation."""
    engine = Glicko2Engine()

    equal = engine.expected_score(
        Glicko2Rating(mu=1500, phi=200, sigma=0.06),
        Glicko2Rating(mu=1500, phi=200, sigma=0.06),
    )
    assert abs(equal - 0.5) < 0.01

    higher_wins = engine.expected_score(
        Glicko2Rating(mu=1800, phi=100, sigma=0.06),
        Glicko2Rating(mu=1200, phi=100, sigma=0.06),
    )
    assert higher_wins > 0.9


def test_win_increases_rating():
    engine = Glicko2Engine(tau=0.5)
    player = Glicko2Rating(mu=1500, phi=200, sigma=0.06)
    opponent = Glicko2Rating(mu=1500, phi=200, sigma=0.06)

    result = engine.rate(player, [opponent], [1.0])
    assert result.mu > 1500


def test_loss_decreases_rating():
    engine = Glicko2Engine(tau=0.5)
    player = Glicko2Rating(mu=1500, phi=200, sigma=0.06)
    opponent = Glicko2Rating(mu=1500, phi=200, sigma=0.06)

    result = engine.rate(player, [opponent], [0.0])
    assert result.mu < 1500
