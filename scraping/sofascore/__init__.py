"""Collection is explicit: importing this package never opens a browser."""

from .config import Profile, RatePolicy, Scope
from .store import Store

__all__ = ["Profile", "RatePolicy", "Scope", "Store"]

