"""stub TrackingConfig (only used inside eval methods we do not invoke)."""
class TrackingConfig:
    def __init__(self, d=None):
        self._d = d or {}
        self.class_range = (d or {}).get("class_range", {}) if isinstance(d, dict) else {}
    @classmethod
    def deserialize(cls, d=None):
        return cls(d)
