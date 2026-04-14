"""Pure Glicko-2 algorithm implementation per Glickman's specification.

Reference: http://www.glicko.net/glicko/glicko2.pdf
"""

from __future__ import annotations

import math
from dataclasses import dataclass

SCALE = 173.7178  # Glicko-2 scaling constant (400 / ln(10))


@dataclass
class Glicko2Rating:
    mu: float = 1500.0
    phi: float = 350.0
    sigma: float = 0.06


class Glicko2Engine:
    """Glicko-2 rating engine."""

    def __init__(
        self,
        tau: float = 0.5,
        epsilon: float = 1e-6,
        max_iterations: int = 100,
        initial_mu: float = 1500.0,
        initial_phi: float = 350.0,
        initial_sigma: float = 0.06,
    ):
        self.tau = tau
        self.epsilon = epsilon
        self.max_iterations = max_iterations
        self.initial_mu = initial_mu
        self.initial_phi = initial_phi
        self.initial_sigma = initial_sigma

    def new_rating(self) -> Glicko2Rating:
        return Glicko2Rating(self.initial_mu, self.initial_phi, self.initial_sigma)

    def rate(
        self,
        player: Glicko2Rating,
        opponents: list[Glicko2Rating],
        outcomes: list[float],
    ) -> Glicko2Rating:
        """Full Glicko-2 update for one rating period.

        Args:
            player: Current player rating.
            opponents: List of opponent ratings.
            outcomes: 1.0 = win, 0.5 = draw, 0.0 = loss for each opponent.

        Returns:
            Updated Glicko2Rating.
        """
        # Step 1: If no games, increase uncertainty only
        if not opponents:
            new_phi = math.sqrt(player.phi**2 + player.sigma**2)
            return Glicko2Rating(player.mu, min(new_phi, self.initial_phi), player.sigma)

        # Step 2: Scale down to Glicko-2 internal scale
        mu, phi, sigma = self._scale_down(player)
        opp_scaled = [self._scale_down(o) for o in opponents]

        # Step 3: Compute v (estimated variance of rating based on game outcomes)
        v = self._compute_v(mu, opp_scaled)

        # Step 4: Compute delta (estimated improvement)
        delta = self._compute_delta(mu, opp_scaled, outcomes, v)

        # Step 5: Determine new sigma (volatility) via Illinois algorithm
        new_sigma = self._determine_sigma(sigma, phi, v, delta)

        # Step 6: Update phi
        phi_star = math.sqrt(phi**2 + new_sigma**2)

        # Step 7: Update phi and mu
        new_phi = 1.0 / math.sqrt(1.0 / phi_star**2 + 1.0 / v)
        new_mu = mu + new_phi**2 * sum(
            self._g(op[1]) * (s - self._E(mu, op[0], op[1]))
            for op, s in zip(opp_scaled, outcomes)
        )

        # Step 8: Scale back up
        return self._scale_up(new_mu, new_phi, new_sigma)

    def expected_score(self, player: Glicko2Rating, opponent: Glicko2Rating) -> float:
        """Win probability from Glicko-2 ratings."""
        mu_p, phi_p, _ = self._scale_down(player)
        mu_o, phi_o, _ = self._scale_down(opponent)
        return self._E(mu_p, mu_o, phi_o)

    # --- Internal methods ---

    def _scale_down(self, rating: Glicko2Rating) -> tuple[float, float, float]:
        return (
            (rating.mu - self.initial_mu) / SCALE,
            rating.phi / SCALE,
            rating.sigma,
        )

    def _scale_up(self, mu: float, phi: float, sigma: float) -> Glicko2Rating:
        return Glicko2Rating(
            mu=mu * SCALE + self.initial_mu,
            phi=phi * SCALE,
            sigma=sigma,
        )

    @staticmethod
    def _g(phi: float) -> float:
        """g(phi) = 1 / sqrt(1 + 3*phi^2 / pi^2)"""
        return 1.0 / math.sqrt(1.0 + 3.0 * phi**2 / math.pi**2)

    def _E(self, mu: float, mu_j: float, phi_j: float) -> float:
        """Expected score E(mu, mu_j, phi_j)."""
        exponent = -self._g(phi_j) * (mu - mu_j)
        exponent = max(-700, min(700, exponent))  # prevent overflow
        return 1.0 / (1.0 + math.exp(exponent))

    def _compute_v(self, mu: float, opponents: list[tuple[float, float, float]]) -> float:
        """Estimated variance of rating based on game outcomes."""
        total = 0.0
        for mu_j, phi_j, _ in opponents:
            g_val = self._g(phi_j)
            e_val = self._E(mu, mu_j, phi_j)
            total += g_val**2 * e_val * (1.0 - e_val)
        return 1.0 / total if total > 0 else 1e10

    def _compute_delta(
        self,
        mu: float,
        opponents: list[tuple[float, float, float]],
        outcomes: list[float],
        v: float,
    ) -> float:
        """Estimated improvement in rating."""
        total = sum(
            self._g(phi_j) * (s - self._E(mu, mu_j, phi_j))
            for (mu_j, phi_j, _), s in zip(opponents, outcomes)
        )
        return v * total

    def _determine_sigma(
        self, sigma: float, phi: float, v: float, delta: float
    ) -> float:
        """Iteratively determine new sigma using Illinois variant of regula falsi.

        This is Step 5 of the Glicko-2 algorithm.
        """
        # Clamp extreme values to prevent overflow
        delta = max(-1e6, min(1e6, delta))
        v = max(1e-10, min(1e10, v))
        phi = max(1e-10, min(1e4, phi))

        a = math.log(sigma**2)
        tau_sq = self.tau**2

        def f(x: float) -> float:
            x = max(-700, min(700, x))
            ex = math.exp(x)
            phi_sq = phi**2
            d_sq = delta**2
            num = ex * (d_sq - phi_sq - v - ex)
            denom = 2.0 * (phi_sq + v + ex) ** 2
            if denom == 0:
                return 0.0
            return num / denom - (x - a) / tau_sq

        # Set initial bounds
        d_sq = delta**2
        phi_sq = phi**2
        if d_sq > phi_sq + v:
            diff = d_sq - phi_sq - v
            if diff > 0:
                b = math.log(diff)
            else:
                b = a - self.tau
        else:
            k = 1
            while f(a - k * self.tau) < 0:
                k += 1
                if k > 100:
                    break
            b = a - k * self.tau

        fa = f(a)
        fb = f(b)

        for _ in range(self.max_iterations):
            if abs(b - a) < self.epsilon:
                break

            denom = fb - fa
            if abs(denom) < 1e-15:
                break
            c = a + (a - b) * fa / denom
            fc = f(c)

            if fc * fb <= 0:
                a = b
                fa = fb
            else:
                fa /= 2.0

            b = c
            fb = fc

        result = math.exp(max(-700, min(700, b / 2.0)))
        return min(result, 0.5)  # cap volatility
