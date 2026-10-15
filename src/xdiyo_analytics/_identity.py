"""Private identity tracking without retaining ordinary fitted objects."""
from weakref import ref


class IdentityRegistry:
    """Track living objects, including unhashable ones; retain nonweakref objects.

    A dead reference is removed only if it is still the entry for that identity.
    Thus an old callback cannot remove a newer object after ID recycling.
    """
    def __init__(self):
        self._weak = {}
        self._strong = {}

    def __contains__(self, value):
        key = id(value)
        weak = self._weak.get(key)
        return (weak is not None and weak() is value) or (key in self._strong and self._strong[key] is value)

    def add(self, value):
        key = id(value)
        def remove(dead):
            if self._weak.get(key) is dead:
                del self._weak[key]
        try:
            self._weak[key] = ref(value, remove)
        except TypeError:
            self._strong[key] = value

    def extend(self, values):
        for value in values:
            self.add(value)

    def __getstate__(self):
        # Serialize live referents, never weakref callbacks or address keys.
        # Pickle memoization preserves aliases with factories/adapters in the
        # same object graph, so reused fitted objects remain detectable.
        values = [value for weak in list(self._weak.values()) if (value := weak()) is not None]
        return values + list(self._strong.values())

    def __setstate__(self, values):
        self.__init__()
        self.extend(values)
