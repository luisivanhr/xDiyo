from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Job:
    path: str
    component: str
    event_id: int | None = None
    team_id: int | None = None
    locale: str = "en"
    reasons: tuple[str, ...] = ()

    def __post_init__(self):
        if not self.path.startswith("/") or ".." in self.path or any(c in self.path for c in "?#\r\n"):
            raise ValueError("Job path must be a relative API path without query or traversal")

    @property
    def key(self) -> str:
        return f"sofascore|v1|{self.locale}|{self.path}"


@dataclass
class Response:
    status: int | None
    payload: Any = None
    headers: dict[str, str] = field(default_factory=dict)
    error: str | None = None
    # Observed navigations in addition to the one reserved before dispatch.
    additional_api_requests: int = 0


class CollectionPaused(RuntimeError):
    """The checkpoint is safe; explicit resumption is needed."""


class BudgetExhausted(CollectionPaused):
    pass

