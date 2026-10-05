import random
import string
import time
from enum import Enum, auto
from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import AppConfig
from detection import DetectionEngine, DetectionItem


class InspectionStep(Enum):
    FRONT_IDLE = auto()
    FRONT_RUNNING = auto()
    FRONT_DONE = auto()
    BACK_RUNNING = auto()
    BACK_DONE = auto()
    BARCODE_POLLING = auto()
    CYCLE_COMPLETE = auto()


class MothersonInspectionCycle:
    """Manages the complete Motherson multi-step inspection workflow."""

    def __init__(self, config: AppConfig, engine: DetectionEngine):
        self.config = config
        self.engine = engine

        # State tracking
        self.step = InspectionStep.FRONT_IDLE
        self.front_timestamp: float = 0.0
        self.back_timestamp: float = 0.0

        # Front Module Results
        self.front_frame: Optional[np.ndarray] = None
        self.front_annotated: Optional[np.ndarray] = None
        self.front_part_present: bool = True
        self.front_scratch_good: bool = True
        self.front_defects: List[str] = []
        self.front_detections: List[DetectionItem] = []

        # Back Module Results
        self.back_frame: Optional[np.ndarray] = None
        self.back_annotated: Optional[np.ndarray] = None
        self.back_part_present: bool = True
        self.back_cap_good: bool = True
        self.back_defect2_good: bool = True
        self.back_defects: List[str] = []
        self.back_detections: List[DetectionItem] = []

        # Barcode Results
        self.barcode_attempts: int = 0
        self.barcode_success: bool = False
        self.barcode_data: str = ""
        self.barcode_warning_banner: bool = False

        # Overall Status
        self.overall_good: bool = False
        self.all_defects: List[str] = []

    def reset_cycle(self):
        """Resets the entire cycle back to initial Front Module state."""
        self.step = InspectionStep.FRONT_IDLE
        self.front_timestamp = 0.0
        self.back_timestamp = 0.0

        self.front_frame = None
        self.front_annotated = None
        self.front_part_present = True
        self.front_scratch_good = True
        self.front_defects = []
        self.front_detections = []

        self.back_frame = None
        self.back_annotated = None
        self.back_part_present = True
        self.back_cap_good = True
        self.back_defect2_good = True
        self.back_defects = []
        self.back_detections = []

        self.barcode_attempts = 0
        self.barcode_success = False
        self.barcode_data = ""
        self.barcode_warning_banner = False

        self.overall_good = False
        self.all_defects = []

    def process_front(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, List[str]]:
        """Processes Front Module (part_presence.pt + scratch.pt)."""
        self.step = InspectionStep.FRONT_RUNNING
        self.front_frame = frame.copy()
        self.front_timestamp = time.time()
        self.front_defects = []

        # 1. Part Presence check on Front
        presence_dets = self.engine.detect_model("part_presence", frame)
        if self.engine.models["part_presence"].is_ready:
            self.front_part_present = len(presence_dets) > 0
        else:
            # Heuristic simulation if model not loaded yet
            self.front_part_present = True

        if not self.front_part_present:
            self.front_defects.append("Front: No Part Present (BAD)")

        # 2. Scratch Model check on Front
        scratch_dets = self.engine.detect_model("scratch", frame)
        self.front_scratch_good = len(scratch_dets) == 0

        if not self.front_scratch_good:
            self.front_defects.append(f"Front: Scratch Detected ({len(scratch_dets)} bbox)")

        self.front_detections = presence_dets + scratch_dets
        self.front_annotated = self.engine.annotate(frame, self.front_detections)

        is_front_good = self.front_part_present and self.front_scratch_good
        self.step = InspectionStep.FRONT_DONE
        return self.front_annotated, is_front_good, self.front_defects

    def process_back(self, frame: np.ndarray) -> Tuple[np.ndarray, bool, List[str]]:
        """Processes Back Module (part_presence.pt + cap.pt + defect_2.pt simultaneously)."""
        self.step = InspectionStep.BACK_RUNNING
        self.back_frame = frame.copy()
        self.back_timestamp = time.time()
        self.back_defects = []

        # 1. Part Presence check on Back
        presence_dets = self.engine.detect_model("part_presence", frame)
        if self.engine.models["part_presence"].is_ready:
            self.back_part_present = len(presence_dets) > 0
        else:
            self.back_part_present = True

        if not self.back_part_present:
            self.back_defects.append("Back: No Part Present (BAD)")

        # 2. Cap Model check (cap / foam -> GREEN, no_cap -> RED)
        cap_dets = self.engine.detect_model("cap", frame)
        no_cap_count = sum(
            1 for d in cap_dets
            if "no_cap" in d.class_name.lower() or "nocap" in d.class_name.lower()
        )
        self.back_cap_good = (no_cap_count == 0)

        if not self.back_cap_good:
            self.back_defects.append(f"Back: No Cap Detected ({no_cap_count} no_cap bbox)")

        # 3. Defect 2 Model check (NG if any bbox)
        defect2_dets = self.engine.detect_model("defect_2", frame)
        self.back_defect2_good = (len(defect2_dets) == 0)

        if not self.back_defect2_good:
            self.back_defects.append(f"Back: Defect 2 Detected ({len(defect2_dets)} bbox)")

        self.back_detections = presence_dets + cap_dets + defect2_dets
        self.back_annotated = self.engine.annotate(frame, self.back_detections)

        is_back_good = (
            self.back_part_present and self.back_cap_good and self.back_defect2_good
        )
        self.step = InspectionStep.BACK_DONE
        return self.back_annotated, is_back_good, self.back_defects

    def process_barcode_attempt(self, frame: np.ndarray) -> Tuple[bool, bool, str]:
        """Runs single barcode inference attempt.
        
        Returns: (success, is_last_attempt, barcode_string)
        """
        self.step = InspectionStep.BARCODE_POLLING
        self.barcode_attempts += 1

        barcode_dets = self.engine.detect_model("barcode", frame)

        # Success if barcode bbox detected or fallback simulation
        if len(barcode_dets) > 0:
            self.barcode_success = True
            item = barcode_dets[0]
            self.barcode_data = item.class_name if item.class_name.isalnum() else self._generate_random_barcode()
        else:
            # Simulation fallback: after 3 attempts in demo, produce random barcode
            # unless models exist and fail
            if not self.engine.models["barcode"].is_ready:
                # Simulate barcode reading after a couple polls
                if self.barcode_attempts >= 2:
                    self.barcode_success = True
                    self.barcode_data = self._generate_random_barcode()
                else:
                    self.barcode_success = False
            else:
                self.barcode_success = False

        if self.barcode_success:
            self.barcode_warning_banner = False
            self.finalize_cycle()
            return True, True, self.barcode_data

        if self.barcode_attempts >= self.config.barcode_max_attempts:
            self.barcode_warning_banner = True
            self.barcode_success = False
            self.barcode_data = "NO BARCODE READ"
            self.finalize_cycle()
            return False, True, self.barcode_data

        return False, False, ""

    def _generate_random_barcode(self) -> str:
        """Generates random barcode string e.g. MTH-84920194."""
        numbers = "".join(random.choices(string.digits, k=8))
        return f"MTH-{numbers}"

    def finalize_cycle(self):
        """Computes final overall result."""
        self.step = InspectionStep.CYCLE_COMPLETE

        self.all_defects = []
        if not self.front_part_present:
            self.all_defects.append("Front Module: No Part Present")
        if not self.front_scratch_good:
            self.all_defects.append("Front Module: Scratch Defect Detected")

        # Include back module defects if front passed
        if self.front_part_present and self.front_scratch_good:
            if not self.back_part_present:
                self.all_defects.append("Back Module: No Part Present")
            if not self.back_cap_good:
                self.all_defects.append("Back Module: No Cap Detected")
            if not self.back_defect2_good:
                self.all_defects.append("Back Module: Defect 2 Detected")

            # Include barcode failure only if back passed and barcode was attempted
            if self.back_part_present and self.back_cap_good and self.back_defect2_good:
                if self.barcode_attempts > 0 and not self.barcode_success:
                    self.all_defects.append("Barcode: Failed to Read Barcode")

        self.overall_good = (len(self.all_defects) == 0)
