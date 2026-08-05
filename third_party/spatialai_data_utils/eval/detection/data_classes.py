"""stub DetectionConfig (only used inside eval methods we do not invoke)."""
class _Cfg:
    def __init__(self, d=None):
        self._d = d or {}
        self.class_range = {}
        if isinstance(d, dict):
            self.class_range = d.get("class_range", {})
    def __getattr__(self, k):
        return self.__dict__.get("_d", {}).get(k) if isinstance(self.__dict__.get("_d"), dict) else None
class DetectionConfig(_Cfg):
    @classmethod
    def deserialize(cls, d=None):
        return cls(d)
