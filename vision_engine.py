from typing import Dict, Optional, Tuple
from config import AppConfig
from detection import DetectionEngine


class VisionEngine:
    """High-level wrapper for Motherson multi-model detection engine."""

    def __init__(self, config: AppConfig):
        self.config = config
        self.engine = DetectionEngine(config)

    @property
    def is_ready(self) -> bool:
        # Ready if at least scratch or cap model is available
        return (
            self.engine.models["scratch"].is_ready
            or self.engine.models["cap"].is_ready
        )

    def reload(self):
        self.engine.reload_all_models()

    def get_model_status(self) -> Dict[str, Tuple[bool, Optional[str]]]:
        return {
            key: (m.is_ready, m.error)
            for key, m in self.engine.models.items()
        }
