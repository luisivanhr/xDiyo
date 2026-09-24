from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Sequence, Tuple


# --- Public score constants ---------------------------------------------------
WIN: float = 1.0
DRAW: float = 0.5
LOSS: float = 0.0

# --- Public-scale defaults (Glicko-style numbers) -----------------------------
MU_DEFAULT: float = 1500.0
PHI_DEFAULT: float = 350.0
SIGMA_DEFAULT: float = 0.06

# --- Glicko-2 hyperparameters -------------------------------------------------
TAU_DEFAULT: float = 1.0        # volatility change constraint
EPSILON_DEFAULT: float = 1e-6   # root-finding tolerance

# --- Scale conversion constant ------------------------------------------------
# 173.7178 = 400 / ln(10); converts between public scale and Glicko-2 scale
SCALE_RATIO: float = 173.7178


Result = Literal[0.0, 0.5, 1.0]  # LOSS, DRAW, WIN for r1



@dataclass(frozen=True)
class Rating:
    """Player rating on the PUBLIC scale (mu≈1500, phi≈350)."""
    mu: float = MU_DEFAULT
    phi: float = PHI_DEFAULT
    sigma: float = SIGMA_DEFAULT

    def __str__(self) -> str:
        return f"Rating(mu={self.mu:.3f}, phi={self.phi:.3f}, sigma={self.sigma:.3f})"


class Glicko2:
    """Glicko-2 rating system with standard update rules."""

    def __init__(
        self,
        *,
        mu0: float = MU_DEFAULT,
        phi0: float = PHI_DEFAULT,
        sigma0: float = SIGMA_DEFAULT,
        tau: float = TAU_DEFAULT,
        epsilon: float = EPSILON_DEFAULT,
    ) -> None:
        self.mu0: float = mu0
        self.phi0: float = phi0
        self.sigma0: float = sigma0
        self.tau: float = tau
        self.epsilon: float = epsilon

    # -- Construction helpers --------------------------------------------------

    def create_rating(self, mu: float | None = None, phi: float | None = None, sigma: float | None = None) -> Rating:
        """Create a rating on the PUBLIC scale."""
        return Rating(
            mu=self.mu0 if mu is None else mu,
            phi=self.phi0 if phi is None else phi,
            sigma=self.sigma0 if sigma is None else sigma,
        )

    # -- Scale transforms ------------------------------------------------------

    def _to_glicko2_scale(self, r: Rating) -> Rating:
        """Convert PUBLIC scale -> Glicko-2 scale (mu', phi'); sigma is unchanged."""
        return Rating(
            mu=(r.mu - self.mu0) / SCALE_RATIO,
            phi=r.phi / SCALE_RATIO,
            sigma=r.sigma,
        )

    def _to_public_scale(self, r: Rating) -> Rating:
        """Convert Glicko-2 scale -> PUBLIC scale."""
        return Rating(
            mu=r.mu * SCALE_RATIO + self.mu0,
            phi=r.phi * SCALE_RATIO,
            sigma=r.sigma,
        )

    # -- Core Glicko-2 functions ----------------------------------------------

    @staticmethod
    def _impact(phi: float) -> float:
        """g(phi): reduces impact as opponent uncertainty (RD) grows."""
        return 1.0 / math.sqrt(1.0 + (3.0 * phi * phi) / (math.pi * math.pi))

    @staticmethod
    def _expected_score(mu: float, mu_op: float, g_op: float) -> float:
        """E = 1 / (1 + exp(-g(phi_op) * (mu - mu_op)))."""
        return 1.0 / (1.0 + math.exp(-g_op * (mu - mu_op)))

    def _solve_new_sigma(self, phi: float, sigma: float, delta: float, v: float) -> float:
        """
        Step 5 root-finding for new volatility sigma'.
        Uses the standard function f(x) with x = ln(sigma'^2).
        """
        delta2 = delta * delta
        alpha = math.log(sigma * sigma)

        def f(x: float) -> float:
            # tmp = phi^2 + v + e^x
            ex = math.exp(x)
            tmp = phi * phi + v + ex
            a = ex * (delta2 - tmp) / (2.0 * tmp * tmp)
            b = (x - alpha) / (self.tau * self.tau)
            return a - b

        # Initial bounds a, b
        a = alpha
        if delta2 > phi * phi + v:
            b = math.log(delta2 - phi * phi - v)
        else:
            k = 1
            while f(alpha - k * self.tau) < 0.0:
                k += 1
            b = alpha - k * self.tau

        f_a = f(a)
        f_b = f(b)

        # Illinois variant of regula falsi
        while abs(b - a) > self.epsilon:
            c = a + (a - b) * f_a / (f_b - f_a)
            f_c = f(c)
            if f_c * f_b < 0.0:
                a, f_a = b, f_b
            else:
                f_a *= 0.5
            b, f_b = c, f_c

        return math.exp(a / 2.0)  # sigma' = exp(x/2)

    # -- Public API ------------------------------------------------------------

    def rate(self, r: Rating, results: Sequence[Tuple[float, Rating]]) -> Rating:
        """
        Update a player's rating given a sequence of (score, opponent_rating) on the PUBLIC scale.

        Args:
            r: player's current rating (PUBLIC scale).
            results: sequence of (s_j, opp_j), where s_j in {WIN, DRAW, LOSS}.

        Returns:
            Updated rating (PUBLIC scale).
        """
        # Step 2: to Glicko-2 scale
        me = self._to_glicko2_scale(r)

        # If no games played: only RD inflation (Step 6), then back to public scale
        if not results:
            phi_star = math.sqrt(me.phi * me.phi + me.sigma * me.sigma)
            return self._to_public_scale(Rating(mu=me.mu, phi=phi_star, sigma=me.sigma))

        # Steps 3–4: accumulate v^{-1} and the numerator for Δ
        inv_v_sum: float = 0.0                    # Σ g_j^2 * E_j * (1 - E_j)
        sum_g_score_diff: float = 0.0             # Σ g_j * (s_j - E_j)

        for s_j, opp_public in results:
            opp = self._to_glicko2_scale(opp_public)
            g_j = self._impact(opp.phi)                          # g(φ_j) using opponent RD
            E_j = self._expected_score(me.mu, opp.mu, g_j)       # E versus opponent j
            inv_v_sum += g_j * g_j * E_j * (1.0 - E_j)
            sum_g_score_diff += g_j * (s_j - E_j)

        v: float = 1.0 / inv_v_sum                                # Step 3: variance
        delta: float = v * sum_g_score_diff                        # Step 4: rating improvement Δ

        # Step 5: new volatility sigma'
        sigma_prime: float = self._solve_new_sigma(phi=me.phi, sigma=me.sigma, delta=delta, v=v)

        # Step 6: pre-period RD inflation
        phi_star: float = math.sqrt(me.phi * me.phi + sigma_prime * sigma_prime)

        # Step 7: posterior update
        phi_prime: float = 1.0 / math.sqrt((1.0 / (phi_star * phi_star)) + (1.0 / v))
        mu_prime: float = me.mu + (phi_prime * phi_prime) * sum_g_score_diff

        # Step 8: back to PUBLIC scale
        return self._to_public_scale(Rating(mu=mu_prime, phi=phi_prime, sigma=sigma_prime))

    def rate_1vs1_result(self, r1: Rating, r2: Rating, *, r1_result: Result) -> Tuple[Rating, Rating]:
        """Update two players; r1_result is from r1’s perspective: 1.0 win, 0.5 draw, 0.0 loss."""
        s1 = float(r1_result)
        s2 = 1.0 - s1 if s1 in (0.0, 1.0) else 0.5
        return self.rate(r1, [(s1, r2)]), self.rate(r2, [(s2, r1)])

    def rate_1vs1_scores(self, r1: Rating, r2: Rating, *, s1: float, s2: float) -> Tuple[Rating, Rating]:
        """Update two players with explicit scores (1,0), (0,1), or (0.5,0.5)."""
        if (s1, s2) not in [(1.0, 0.0), (0.0, 1.0), (0.5, 0.5)]:
            raise ValueError("Scores must be (1,0), (0,1), or (0.5,0.5).")
        return self.rate(r1, [(s1, r2)]), self.rate(r2, [(s2, r1)])

    # Back-compat thin wrapper (same behavior as before):
    def rate_1vs1(self, r1: Rating, r2: Rating, *, drawn: bool = False) -> Tuple[Rating, Rating]:
        """Legacy: assumes r1 wins unless drawn=True."""
        return self.rate_1vs1_result(r1, r2, r1_result=0.5 if drawn else 1.0)
    
    def advance_periods(self, r: Rating, k_eff: float, *, cap_to_prior: bool = True) -> Rating:
        """Inflate RD over k_eff effective idle periods (can be fractional)."""
        if k_eff <= 0:
            return r
        ri = self._to_glicko2_scale(r)
        phi_new = math.sqrt(ri.phi * ri.phi + k_eff * ri.sigma * ri.sigma)
        r_new = self._to_public_scale(Rating(mu=ri.mu, phi=phi_new, sigma=ri.sigma))
        if cap_to_prior and r_new.phi > self.phi0:  # e.g., 350 public
            r_new = Rating(mu=r_new.mu, phi=self.phi0, sigma=r_new.sigma)
        return r_new

    def quality_1vs1(self, r1: Rating, r2: Rating) -> float:
        """
        Symmetric match quality in [0, 1]; higher means more evenly matched.
        Uses each *opponent's* RD in g(·), which is the correct convention.
        """
        r1_g = self._impact(self._to_glicko2_scale(r2).phi)  # use r2's RD
        r2_g = self._impact(self._to_glicko2_scale(r1).phi)  # use r1's RD

        # Expectations on the PUBLIC scale but with correct g(φ_opponent)
        e1 = self._expected_score(
            mu=(r1.mu - self.mu0) / SCALE_RATIO,
            mu_op=(r2.mu - self.mu0) / SCALE_RATIO,
            g_op=r1_g,
        )
        e2 = self._expected_score(
            mu=(r2.mu - self.mu0) / SCALE_RATIO,
            mu_op=(r1.mu - self.mu0) / SCALE_RATIO,
            g_op=r2_g,
        )
        expected_avg = 0.5 * (e1 + e2)
        return 2.0 * (0.5 - abs(0.5 - expected_avg))
