import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    path: Path
    confidence: float = 0.25
    imgsz: int = 640


@dataclass
class AppConfig:
    company_name: str = "MOTHERSON"
    app_name: str = "Automotive Visual Inspection Platform"

    logo_path: Optional[Path] = Path("logo.jpeg")

    # Storage paths
    @property
    def raw_images_dir(self) -> Path:
        return Path("images") / "raw"

    @property
    def ng_images_dir(self) -> Path:
        return Path("images") / "ng"

    @property
    def app_log_file(self) -> Path:
        return Path("app_log.txt")

    # Camera settings (Baumer only)
    active_camera: str = "Baumer"
    camera_trigger_mode: str = "software"  # "software" or "continuous"
    live_refresh_ms: int = 33

    baumer_width: int = 3072
    baumer_height: int = 2048
    baumer_offset_x: int = 0
    baumer_offset_y: int = 0
    baumer_exposure_time: float = 17000.0
    baumer_gain: float = 7
    baumer_gamma: float = 1.15
    baumer_get_image_timeout_ms: int = 1000
    baumer_watchdog_no_frame_sec: int = 300

    # Per-Model Configuration Settings
    scratch_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(Path(r"model\motherson\scratch.pt"), confidence=0.6, imgsz=1024)
    )
    cap_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(Path(r"model\motherson\cap.pt"), confidence=0.4, imgsz=640)
    )
    defect_2_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(Path(r"model\motherson\defect_2.pt"), confidence=0.6, imgsz=1024)
    )
    part_presence_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(Path(r"model\motherson\presence.pt"), confidence=0.25, imgsz=640)
    )
    barcode_config: ModelConfig = field(
        default_factory=lambda: ModelConfig(Path(r"model\motherson\barcode.pt"), confidence=0.25, imgsz=640)
    )

    # Timing and Workflow Settings
    front_to_back_timeout_sec: float = 5
    barcode_initial_delay_ms: int = 5000
    barcode_poll_interval_ms: int = 1000
    barcode_max_attempts: int = 10

    # ROI settings
    roi_x: int = 0
    roi_y: int = 0
    roi_w: int = 9999
    roi_h: int = 9999


def ensure_output_dirs(config: AppConfig) -> None:
    config.raw_images_dir.mkdir(parents=True, exist_ok=True)
    config.ng_images_dir.mkdir(parents=True, exist_ok=True)
