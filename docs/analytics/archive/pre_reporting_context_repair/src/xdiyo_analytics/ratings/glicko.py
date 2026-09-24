"""Snapshot adapter around the existing utils.glicko_rating numerical engine.

Reference: https://www.glicko.net/glicko/glicko2.pdf
Public rating/rd and internal mu/phi are named separately. This module does not
choose football outcomes, calendar periods or prediction cutoffs. All updates
use the existing engine, preserving its public-scale arithmetic and defaults.
"""

from dataclasses import dataclass
import math

from utils.glicko_rating import (
    Glicko2 as _LegacyGlicko2, Rating as _LegacyRating,
    SCALE_RATIO, TAU_DEFAULT, EPSILON_DEFAULT,
)
from typing import Mapping, Protocol, Sequence

SCALE = SCALE_RATIO


class PairwiseRatingEngine(Protocol):
    """Minimal engine contract; snapshots may also be produced by other models."""

    def initial_state(self) -> dict[str, float]: ...

    def update_period(self, state: Mapping[str, float],
                      games: Sequence[tuple[float, Mapping[str, float]]]) -> dict[str, float]: ...


def glicko_state(rating=1500.0, rd=350.0, sigma=0.06):
    """Construct a state with public rating/rd and internal mu/phi coordinates."""
    if not all(math.isfinite(x) for x in (rating, rd, sigma)) or rd <= 0 or sigma <= 0:
        raise ValueError("Rating must be finite; rd and sigma must be finite and positive.")
    return {"rating": float(rating), "rd": float(rd), "sigma": float(sigma),
            "mu": (rating - 1500) / SCALE, "phi": rd / SCALE}


@dataclass(frozen=True)
class Glicko2:
    initial_rating: float = 1500.0
    initial_rd: float = 350.0
    initial_sigma: float = 0.06
    tau: float = TAU_DEFAULT
    tolerance: float = EPSILON_DEFAULT

    def __post_init__(self):
        self.initial_state()
        if not all(math.isfinite(x) and x > 0 for x in (self.tau, self.tolerance)):
            raise ValueError("tau and tolerance must be finite and positive.")

    def initial_state(self):
        return glicko_state(self.initial_rating, self.initial_rd, self.initial_sigma)

    def _engine(self):
        return _LegacyGlicko2(
            mu0=self.initial_rating, phi0=self.initial_rd,
            sigma0=self.initial_sigma, tau=self.tau, epsilon=self.tolerance,
        )

    @staticmethod
    def _public(state):
        return _LegacyRating(state["rating"], state["rd"], state["sigma"])

    def update_period(self, state, games):
        """Use the legacy arithmetic with pre-period opponent states.

        Public rating/rd/sigma outputs exactly match utils.glicko_rating for
        identical parameters and inputs. mu/phi are additional standard internal
        coordinates, not the legacy Rating object's public-scale field names.
        Empty periods increase uncertainty once; replay inserts none implicitly.
        """
        observations = []
        for score, opponent in games:
            if not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Pairwise outcomes must lie between zero and one.")
            observations.append((score, self._public(opponent)))
        updated = self._engine().rate(self._public(state), observations)
        return glicko_state(updated.mu, updated.phi, updated.sigma)

    def advance_periods(self, state, periods, *, cap_to_prior=True):
        """Explicit legacy-compatible idle-period inflation, never automatic.

        A caller that models calendar inactivity must choose its period mapping;
        do not count the active update period twice. Default cap matches legacy.
        """
        if not math.isfinite(periods) or periods < 0:
            raise ValueError("Idle periods must be finite and nonnegative.")
        updated = self._engine().advance_periods(
            self._public(state), periods, cap_to_prior=cap_to_prior,
        )
        return glicko_state(updated.mu, updated.phi, updated.sigma)
