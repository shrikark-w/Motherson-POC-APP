import os
import sys
import time
import random
from pathlib import Path
from typing import Optional
import numpy as np
import cv2
from PyQt5.QtCore import QPoint, QRect, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from baumer_cam import BaumerCameraThread
from config import AppConfig, ensure_output_dirs
from detection import DetectionItem
from inspection_state import InspectionStep, MothersonInspectionCycle
from vision_engine import VisionEngine


class ROISelectLabel(QLabel):
    roi_selected = pyqtSignal(int, int, int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.drawing_mode = False
        self.start_pos = None
        self.end_pos = None
        self.original_size = None

    def set_drawing_mode(self, enabled: bool):
        self.drawing_mode = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.unsetCursor()
        self.start_pos = None
        self.end_pos = None
        self.update()

    def map_to_original(self, label_pos):
        if not self.original_size or self.pixmap() is None or self.pixmap().isNull():
            return label_pos.x(), label_pos.y()

        L_w = self.width()
        L_h = self.height()
        I_w, I_h = self.original_size

        scale = min(L_w / I_w, L_h / I_h)
        if scale <= 0:
            return label_pos.x(), label_pos.y()
        S_w = I_w * scale
        S_h = I_h * scale

        offset_x = (L_w - S_w) / 2
        offset_y = (L_h - S_h) / 2

        x_I = (label_pos.x() - offset_x) / scale
        y_I = (label_pos.y() - offset_y) / scale

        x_clamped = max(0, min(int(x_I), I_w))
        y_clamped = max(0, min(int(y_I), I_h))

        return x_clamped, y_clamped

    def mousePressEvent(self, event):
        if self.drawing_mode and event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.end_pos = event.pos()
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.drawing_mode and self.start_pos is not None:
            self.end_pos = event.pos()
            self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.drawing_mode and self.start_pos is not None and event.button() == Qt.LeftButton:
            self.end_pos = event.pos()
            x1, y1 = self.map_to_original(self.start_pos)
            x2, y2 = self.map_to_original(self.end_pos)

            rx = min(x1, x2)
            ry = min(y1, y2)
            rw = abs(x2 - x1)
            rh = abs(y2 - y1)

            if rw > 2 and rh > 2:
                self.roi_selected.emit(rx, ry, rw, rh)

            self.set_drawing_mode(False)
        else:
            super().mouseReleaseEvent(event)

    def paintEvent(self, event):
        super().paintEvent(event)
        if self.drawing_mode and self.start_pos is not None and self.end_pos is not None:
            painter = QPainter(self)
            pen = QPen(Qt.yellow, 2, Qt.SolidLine)
            painter.setPen(pen)
            rect = QRect(self.start_pos, self.end_pos)
            painter.drawRect(rect)
            painter.end()


class NoWheelComboBox(QComboBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSlider(QSlider):
    def wheelEvent(self, event):
        event.ignore()


class ResultDialog(QDialog):
    """Big prominent final result pop-up modal."""

    def __init__(self, is_good: bool, barcode_data: str, defects: list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MOTHERSON Inspection Result")
        self.resize(550, 420)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)

        # Header Banner
        header = QLabel("GOOD PART" if is_good else "NOT GOOD (NG)")
        header.setAlignment(Qt.AlignCenter)
        header.setFont(QFont("Trebuchet MS", 22, QFont.Bold))

        if is_good:
            header.setStyleSheet(
                "background-color: #10b981; color: #ffffff; padding: 18px; border-radius: 12px;"
            )
        else:
            header.setStyleSheet(
                "background-color: #ef4444; color: #ffffff; padding: 18px; border-radius: 12px;"
            )

        layout.addWidget(header)

        # Barcode Data Box
        barcode_frame = QFrame()
        barcode_frame.setStyleSheet(
            "background-color: #0f172a; border: 1px solid #334155; border-radius: 10px; padding: 10px;"
        )
        barcode_layout = QVBoxLayout(barcode_frame)

        bc_title = QLabel("INFERRED BARCODE DATA")
        bc_title.setStyleSheet("color: #94a3b8; font-size: 11px; font-weight: bold;")
        bc_val = QLabel(barcode_data if barcode_data else "NO BARCODE READ")
        bc_val.setStyleSheet("color: #38bdf8; font-size: 18px; font-family: 'Courier New'; font-weight: bold;")

        barcode_layout.addWidget(bc_title)
        barcode_layout.addWidget(bc_val)
        layout.addWidget(barcode_frame)

        # Defects / Details Breakdown Box
        details_frame = QFrame()
        details_frame.setStyleSheet(
            "background-color: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 10px;"
        )
        details_layout = QVBoxLayout(details_frame)

        det_title = QLabel("INSPECTION STATUS DETAILS")
        det_title.setStyleSheet("color: #cbd5e1; font-size: 12px; font-weight: bold;")
        details_layout.addWidget(det_title)

        if is_good:
            msg = QLabel("✓ All front & back inspection checks passed successfully.")
            msg.setStyleSheet("color: #10b981; font-size: 13px;")
            details_layout.addWidget(msg)
        else:
            for defect in defects:
                d_lbl = QLabel(f"• {defect}")
                d_lbl.setStyleSheet("color: #f87171; font-size: 13px; font-weight: bold;")
                d_lbl.setWordWrap(True)
                details_layout.addWidget(d_lbl)

        layout.addWidget(details_frame, 1)

        # Action Button: Continue with next part
        btn_continue = QPushButton("Continue with Next Part")
        btn_continue.setStyleSheet(
            """
            QPushButton {
                background-color: #0ea5e9;
                color: #ffffff;
                border: none;
                border-radius: 12px;
                padding: 14px;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0284c7;
            }
            """
        )
        btn_continue.clicked.connect(self.accept)
        layout.addWidget(btn_continue)


class MothersonUI(QMainWindow):
    def __init__(self, config: AppConfig, detector: Optional[VisionEngine] = None):
        super().__init__()
        self.config = config
        self.setWindowTitle(f"{self.config.company_name} - {self.config.app_name}")
        self.resize(1480, 920)

        ensure_output_dirs(self.config)

        self.good_count = 0
        self.bad_count = 0
        self.total_triggers = 0
        self.last_frame = None
        self.pending_inspection_action: Optional[str] = None

        self.detector = detector if detector is not None else VisionEngine(self.config)
        self.cycle = MothersonInspectionCycle(self.config, self.detector.engine)

        self.cam_thread: Optional[BaumerCameraThread] = None

        # Timers
        self.front_auto_timer = QTimer(self)
        self.front_auto_timer.setSingleShot(True)
        self.front_auto_timer.timeout.connect(self.on_front_auto_timeout)

        self.barcode_delay_timer = QTimer(self)
        self.barcode_delay_timer.setSingleShot(True)
        self.barcode_delay_timer.timeout.connect(self.start_barcode_polling)

        self.barcode_poll_timer = QTimer(self)
        self.barcode_poll_timer.timeout.connect(self.on_barcode_poll_tick)

        self._build_ui()
        self._connect_camera()
        self.update_button_states()

    def _build_ui(self):
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f4f7fb;
            }
            QWidget {
                font-family: 'Trebuchet MS';
                color: #0f172a;
            }
            QFrame#leftPanel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #0f172a, stop:1 #1e293b);
                border-radius: 18px;
            }
            QLabel#title {
                color: #f8fafc;
                font-size: 24px;
                font-weight: 700;
            }
            QLabel#subtitle {
                color: #cbd5e1;
                font-size: 13px;
            }
            QFrame#card {
                background-color: #1e293b;
                border: 1px solid #334155;
                border-radius: 14px;
            }
            QLabel#cardLabel {
                color: #94a3b8;
                font-size: 12px;
            }
            QLabel#cardValue {
                color: #f8fafc;
                font-size: 26px;
                font-weight: 700;
            }
            QLabel#logoRoom {
                border: none;
                border-radius: 14px;
                color: #e2e8f0;
                font-size: 14px;
                font-weight: 600;
            }
            QPushButton {
                background-color: #0ea5e9;
                color: #ffffff;
                border: none;
                border-radius: 12px;
                padding: 12px 16px;
                font-size: 14px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #0284c7;
            }
            QPushButton:pressed {
                background-color: #0369a1;
            }
            QPushButton:disabled {
                background-color: #334155;
                color: #64748b;
            }
            QPushButton#emeraldBtn {
                background-color: #10b981;
            }
            QPushButton#emeraldBtn:hover {
                background-color: #059669;
            }
            QPushButton#emeraldBtn:disabled {
                background-color: #334155;
                color: #64748b;
            }
            QPushButton#roseBtn {
                background-color: #ef4444;
            }
            QPushButton#roseBtn:hover {
                background-color: #dc2626;
            }
            QFrame#feedFrame {
                background-color: #ffffff;
                border: 1px solid #dbe4f0;
                border-radius: 18px;
            }
            QLabel#status {
                color: #334155;
                font-size: 13px;
            }
            QLabel#feedLabel {
                border-radius: 14px;
                background-color: #e2e8f0;
                color: #475569;
                font-size: 16px;
            }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background-color: #0f172a;
                border: 1px solid #334155;
                border-radius: 8px;
                padding: 4px 8px;
                font-size: 12px;
                color: #f8fafc;
            }
            """
        )

        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(16, 16, 16, 16)
        root_layout.setSpacing(16)

        # Top Bar
        top_bar = QFrame()
        top_bar.setObjectName("feedFrame")
        top_bar_layout = QHBoxLayout(top_bar)
        top_bar_layout.setContentsMargins(14, 12, 14, 12)
        top_bar_layout.setSpacing(16)

        cam_label = QLabel("Camera Device: Baumer (Industrial)")
        cam_label.setStyleSheet("font-weight: bold; color: #0ea5e9;")
        top_bar_layout.addWidget(cam_label)

        mode_label = QLabel("Trigger Mode")
        mode_label.setStyleSheet("font-weight: bold; color: #475569;")
        top_bar_layout.addWidget(mode_label)

        self.mode_selector = NoWheelComboBox()
        self.mode_selector.addItem("Software Trigger Mode", "software")
        self.mode_selector.addItem("Continuous Live Feed", "continuous")
        self.mode_selector.currentIndexChanged.connect(self.on_mode_changed)
        top_bar_layout.addWidget(self.mode_selector)

        top_bar_layout.addStretch(1)

        self.phase_badge = QLabel("PHASE: FRONT MODULE READY")
        self.phase_badge.setStyleSheet(
            "background-color: #0ea5e9; color: #ffffff; padding: 6px 14px; border-radius: 8px; font-weight: bold;"
        )
        top_bar_layout.addWidget(self.phase_badge)

        self.btn_reset_cycle = QPushButton("Reset Cycle")
        self.btn_reset_cycle.setObjectName("roseBtn")
        self.btn_reset_cycle.clicked.connect(self.on_reset_cycle_clicked)
        top_bar_layout.addWidget(self.btn_reset_cycle)

        root_layout.addWidget(top_bar)

        # Content Layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(16)

        # Left Panel (Controls & Settings)
        left_panel = QFrame()
        left_panel.setObjectName("leftPanel")
        left_panel.setFixedWidth(380)

        left_outer_layout = QVBoxLayout(left_panel)
        left_outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")

        scroll_widget = QWidget()
        left_layout = QVBoxLayout(scroll_widget)
        left_layout.setContentsMargins(20, 20, 20, 20)
        left_layout.setSpacing(12)
        scroll.setWidget(scroll_widget)
        left_outer_layout.addWidget(scroll)

        # Header Title Card
        title_card = QFrame()
        title_card.setStyleSheet(
            "background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0ea5e9, stop:1 #0284c7); "
            "border-radius: 12px; padding: 4px;"
        )
        title_card_layout = QVBoxLayout(title_card)
        title_card_layout.setContentsMargins(10, 10, 10, 10)
        title_card_layout.setSpacing(2)

        title = QLabel(self.config.company_name)
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("color: #ffffff; font-size: 26px; font-weight: 900; letter-spacing: 2px;")

        subtitle = QLabel(self.config.app_name)
        subtitle.setAlignment(Qt.AlignCenter)
        subtitle.setStyleSheet("color: #e0f2fe; font-size: 11px; font-weight: 600;")

        title_card_layout.addWidget(title)
        title_card_layout.addWidget(subtitle)
        left_layout.addWidget(title_card)

        # Metric Statistics Cards
        self.good_value = self._add_metric_card(left_layout, "Good Parts", "0")
        self.bad_value = self._add_metric_card(left_layout, "Bad Parts (NG)", "0")
        self.bad_percent_value = self._add_metric_card(left_layout, "Fail Rate (%)", "0.0%")

        # Inspection Step Buttons Card
        step_card = QFrame()
        step_card.setObjectName("card")
        step_layout = QVBoxLayout(step_card)
        step_layout.setContentsMargins(14, 12, 14, 12)
        step_layout.setSpacing(10)

        step_header = QLabel("MOTHERSON WORKFLOW CONTROLS")
        step_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #0ea5e9;")
        step_layout.addWidget(step_header)

        self.btn_inspect_front = QPushButton("Inspect Front Module")
        self.btn_inspect_front.setObjectName("emeraldBtn")
        self.btn_inspect_front.clicked.connect(self.on_inspect_front_clicked)
        step_layout.addWidget(self.btn_inspect_front)

        self.btn_inspect_back = QPushButton("Inspect Back Module")
        self.btn_inspect_back.clicked.connect(self.on_inspect_back_clicked)
        step_layout.addWidget(self.btn_inspect_back)

        btn_row = QHBoxLayout()
        self.trigger_btn = QPushButton("Soft Trigger")
        self.trigger_btn.clicked.connect(self.on_software_trigger)
        btn_row.addWidget(self.trigger_btn)

        self.infer_static_btn = QPushButton("Infer Local Image")
        self.infer_static_btn.clicked.connect(self.on_infer_local_image)
        btn_row.addWidget(self.infer_static_btn)
        step_layout.addLayout(btn_row)

        left_layout.addWidget(step_card)

        # Model Config Tuning Card
        model_card = QFrame()
        model_card.setObjectName("card")
        model_layout = QVBoxLayout(model_card)
        model_layout.setContentsMargins(14, 12, 14, 12)
        model_layout.setSpacing(8)

        model_header = QLabel("MODEL PARAMETERS CONFIG")
        model_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #0ea5e9;")
        model_layout.addWidget(model_header)

        mod_sel_row = QHBoxLayout()
        mod_lbl = QLabel("Target Model:")
        mod_lbl.setStyleSheet("font-weight: bold; color: #cbd5e1; font-size: 11px;")
        self.model_key_selector = NoWheelComboBox()
        self.model_key_selector.addItems(["scratch", "cap", "defect_2", "part_presence", "barcode"])
        self.model_key_selector.currentTextChanged.connect(self.on_model_key_changed)
        mod_sel_row.addWidget(mod_lbl, 3)
        mod_sel_row.addWidget(self.model_key_selector, 7)
        model_layout.addLayout(mod_sel_row)

        conf_row, self.conf_slider, self.conf_spin = self._create_slider_spinbox_row(
            "Confidence:", 0.01, 1.00, 0.25, decimals=2, step=0.01
        )
        model_layout.addLayout(conf_row)

        imgsz_row = QHBoxLayout()
        imgsz_lbl = QLabel("Img Size:")
        imgsz_lbl.setStyleSheet("font-weight: bold; color: #cbd5e1; font-size: 11px;")
        self.imgsz_selector = NoWheelComboBox()
        self.imgsz_selector.addItems(["320", "416", "512", "640", "800", "1024"])
        self.imgsz_selector.setCurrentText("640")
        imgsz_row.addWidget(imgsz_lbl, 3)
        imgsz_row.addWidget(self.imgsz_selector, 7)
        model_layout.addLayout(imgsz_row)

        self.btn_apply_model_config = QPushButton("Apply Model Setting")
        self.btn_apply_model_config.setObjectName("emeraldBtn")
        self.btn_apply_model_config.clicked.connect(self.on_apply_model_config)
        model_layout.addWidget(self.btn_apply_model_config)

        left_layout.addWidget(model_card)

        # Camera settings card
        camera_card = QFrame()
        camera_card.setObjectName("card")
        camera_layout = QVBoxLayout(camera_card)
        camera_layout.setContentsMargins(14, 12, 14, 12)
        camera_layout.setSpacing(8)

        camera_header = QLabel("BAUMER CAMERA PARAMETERS")
        camera_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #0ea5e9;")
        camera_layout.addWidget(camera_header)

        exp_row, self.exposure_slider, self.exposure_spin = self._create_slider_spinbox_row(
            "Exposure:", 100, 100000, self.config.baumer_exposure_time, decimals=0, step=100
        )
        camera_layout.addLayout(exp_row)

        gain_row, self.gain_slider, self.gain_spin = self._create_slider_spinbox_row(
            "Gain:", 0.0, 30.0, self.config.baumer_gain, decimals=1, step=0.1
        )
        camera_layout.addLayout(gain_row)

        self.apply_camera_btn = QPushButton("Apply Camera Parameters")
        self.apply_camera_btn.setObjectName("emeraldBtn")
        self.apply_camera_btn.clicked.connect(self.on_apply_camera_settings)
        camera_layout.addWidget(self.apply_camera_btn)

        left_layout.addWidget(camera_card)

        # ROI Tuning Card
        roi_card = QFrame()
        roi_card.setObjectName("card")
        roi_layout = QVBoxLayout(roi_card)
        roi_layout.setContentsMargins(14, 12, 14, 12)
        roi_layout.setSpacing(8)

        roi_header = QLabel("REGION OF INTEREST (ROI)")
        roi_header.setStyleSheet("font-weight: bold; font-size: 11px; color: #0ea5e9;")
        roi_layout.addWidget(roi_header)

        roi_grid = QGridLayout()
        roi_grid.setSpacing(8)

        roi_grid.addWidget(QLabel("ROI X:"), 0, 0)
        self.roi_x_spin = NoWheelSpinBox()
        self.roi_x_spin.setRange(0, 9999)
        self.roi_x_spin.setValue(self.config.roi_x)
        roi_grid.addWidget(self.roi_x_spin, 0, 1)

        roi_grid.addWidget(QLabel("ROI Y:"), 0, 2)
        self.roi_y_spin = NoWheelSpinBox()
        self.roi_y_spin.setRange(0, 9999)
        self.roi_y_spin.setValue(self.config.roi_y)
        roi_grid.addWidget(self.roi_y_spin, 0, 3)

        roi_grid.addWidget(QLabel("Width:"), 1, 0)
        self.roi_w_spin = NoWheelSpinBox()
        self.roi_w_spin.setRange(1, 9999)
        self.roi_w_spin.setValue(self.config.roi_w)
        roi_grid.addWidget(self.roi_w_spin, 1, 1)

        roi_grid.addWidget(QLabel("Height:"), 1, 2)
        self.roi_h_spin = NoWheelSpinBox()
        self.roi_h_spin.setRange(1, 9999)
        self.roi_h_spin.setValue(self.config.roi_h)
        roi_grid.addWidget(self.roi_h_spin, 1, 3)

        roi_layout.addLayout(roi_grid)

        self.draw_roi_btn = QPushButton("Draw ROI on Feed")
        self.draw_roi_btn.setCheckable(True)
        self.draw_roi_btn.clicked.connect(self.on_draw_roi_clicked)
        roi_layout.addWidget(self.draw_roi_btn)

        left_layout.addWidget(roi_card)

        # Status Summaries
        self.camera_label = QLabel("Camera Status: Connecting...")
        self.camera_label.setObjectName("subtitle")
        left_layout.addWidget(self.camera_label)

        left_layout.addStretch(1)

        # Right Panel (Feed & Results)
        right_panel = QFrame()
        right_panel.setObjectName("feedFrame")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(14, 14, 14, 14)
        right_layout.setSpacing(10)

        self.feed_header = QLabel("Live Feed / Inspection Preview")
        self.feed_header.setFont(QFont("Trebuchet MS", 16, QFont.Bold))
        right_layout.addWidget(self.feed_header)

        # Warning Banner for Barcode
        self.warning_banner = QLabel("⚠️ PLEASE ATTACH THE BARCODE")
        self.warning_banner.setAlignment(Qt.AlignCenter)
        self.warning_banner.setStyleSheet(
            "background-color: #ef4444; color: #ffffff; font-size: 16px; font-weight: bold; padding: 10px; border-radius: 8px;"
        )
        self.warning_banner.setVisible(False)
        right_layout.addWidget(self.warning_banner)

        self.feed_label = ROISelectLabel()
        self.feed_label.setObjectName("feedLabel")
        self.feed_label.setAlignment(Qt.AlignCenter)
        self.feed_label.setMinimumSize(900, 640)
        self.feed_label.roi_selected.connect(self.on_interactive_roi_selected)
        right_layout.addWidget(self.feed_label, 1)

        self.status_label = QLabel("Status: Ready for Front Module Inspection")
        self.status_label.setObjectName("status")
        right_layout.addWidget(self.status_label)

        content_layout.addWidget(left_panel)
        content_layout.addWidget(right_panel, 1)

        root_layout.addLayout(content_layout, 1)
        self.setCentralWidget(root)

    def _add_metric_card(self, parent_layout, label_text, initial_value) -> QLabel:
        card = QFrame()
        card.setObjectName("card")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        lbl = QLabel(label_text)
        lbl.setObjectName("cardLabel")
        val = QLabel(initial_value)
        val.setObjectName("cardValue")

        layout.addWidget(lbl)
        layout.addWidget(val)
        parent_layout.addWidget(card)

        return val

    def _create_slider_spinbox_row(
        self, label_text, min_val, max_val, init_val, decimals=2, step=0.01
    ):
        row = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet("font-weight: bold; color: #cbd5e1; font-size: 11px;")

        slider = NoWheelSlider(Qt.Horizontal)
        spinbox = NoWheelDoubleSpinBox() if decimals > 0 else NoWheelSpinBox()

        if decimals > 0:
            multiplier = 10**decimals
            slider.setRange(int(min_val * multiplier), int(max_val * multiplier))
            slider.setValue(int(init_val * multiplier))
            slider.setSingleStep(int(step * multiplier))

            spinbox.setRange(min_val, max_val)
            spinbox.setValue(init_val)
            spinbox.setDecimals(decimals)
            spinbox.setSingleStep(step)

            slider.valueChanged.connect(lambda v: spinbox.setValue(v / multiplier))
            spinbox.valueChanged.connect(lambda v: slider.setValue(int(v * multiplier)))
        else:
            slider.setRange(int(min_val), int(max_val))
            slider.setValue(int(init_val))
            slider.setSingleStep(int(step))

            spinbox.setRange(int(min_val), int(max_val))
            spinbox.setValue(int(init_val))
            spinbox.setSingleStep(int(step))

            slider.valueChanged.connect(spinbox.setValue)
            spinbox.valueChanged.connect(slider.setValue)

        row.addWidget(label, 3)
        row.addWidget(slider, 5)
        row.addWidget(spinbox, 4)
        return row, slider, spinbox

    def _load_logo_if_available(self):
        if self.config.logo_path and Path(self.config.logo_path).exists():
            pix = QPixmap(str(self.config.logo_path))
            if not pix.isNull():
                scaled = pix.scaled(
                    200,
                    65,
                    Qt.KeepAspectRatio,
                    Qt.SmoothTransformation,
                )
                self.logo_room.setPixmap(scaled)
                return
        self.logo_room.setText("MOTHERSON")

    def _connect_camera(self):
        self.cam_thread = BaumerCameraThread(self.config, self)
        self.cam_thread.frame_captured.connect(self.on_frame_captured)
        self.cam_thread.camera_connected.connect(
            lambda: self.camera_label.setText("Camera: Baumer Connected")
        )
        self.cam_thread.camera_disconnected.connect(
            lambda: self.camera_label.setText("Camera: Baumer Disconnected (Demo Mode)")
        )
        self.cam_thread.start()

    def update_button_states(self):
        """Manages greyed-out / enabled states of Inspect Front and Inspect Back buttons."""
        step = self.cycle.step

        if step == InspectionStep.FRONT_IDLE:
            self.btn_inspect_front.setEnabled(True)
            self.btn_inspect_back.setEnabled(False)  # Greyed out
            self.phase_badge.setText("PHASE 1: FRONT MODULE READY")
            self.phase_badge.setStyleSheet(
                "background-color: #0ea5e9; color: #ffffff; padding: 6px 14px; border-radius: 8px; font-weight: bold;"
            )
        elif step == InspectionStep.FRONT_DONE:
            self.btn_inspect_front.setEnabled(True)
            self.btn_inspect_back.setEnabled(True)  # Activated after Front completes!
            self.phase_badge.setText("PHASE 1 COMPLETE -> PRESS INSPECT BACK")
            self.phase_badge.setStyleSheet(
                "background-color: #10b981; color: #ffffff; padding: 6px 14px; border-radius: 8px; font-weight: bold;"
            )
        elif step in (InspectionStep.BACK_RUNNING, InspectionStep.BACK_DONE, InspectionStep.BARCODE_POLLING):
            self.btn_inspect_front.setEnabled(False)  # Greyed out when Back module starts!
            self.btn_inspect_back.setEnabled(False)  # Greyed out during processing
            self.phase_badge.setText("PHASE 2: BACK MODULE & BARCODE PROCESSING")
            self.phase_badge.setStyleSheet(
                "background-color: #f59e0b; color: #ffffff; padding: 6px 14px; border-radius: 8px; font-weight: bold;"
            )
        elif step == InspectionStep.CYCLE_COMPLETE:
            self.btn_inspect_front.setEnabled(False)
            self.btn_inspect_back.setEnabled(False)
            self.phase_badge.setText("PHASE: INSPECTION COMPLETE")
            self.phase_badge.setStyleSheet(
                "background-color: #8b5cf6; color: #ffffff; padding: 6px 14px; border-radius: 8px; font-weight: bold;"
            )

    def on_frame_captured(self, frame):
        self.last_frame = frame.copy()

        if self.pending_inspection_action == "front":
            self.pending_inspection_action = None
            self.run_front_inspection(frame)
        elif self.pending_inspection_action == "back":
            self.pending_inspection_action = None
            self.run_back_inspection(frame)
        elif self.config.camera_trigger_mode == "continuous" and self.cycle.step in (InspectionStep.FRONT_IDLE, InspectionStep.FRONT_DONE):
            self.display_image(frame)

    def display_image(self, img: np.ndarray):
        if img is None:
            return

        h, w, c = img.shape
        bytes_per_line = c * w
        q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format_BGR888)
        pix = QPixmap.fromImage(q_img)

        self.feed_label.original_size = (w, h)
        scaled_pix = pix.scaled(
            self.feed_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.feed_label.setPixmap(scaled_pix)

    def get_current_frame(self) -> np.ndarray:
        if self.last_frame is not None:
            return self.last_frame.copy()

        # Generate mock test pattern frame if camera not connected
        mock = np.zeros((1080, 1920, 3), dtype=np.uint8)
        mock[:] = (30, 25, 20)
        cv2.putText(
            mock,
            "MOTHERSON TEST FRAME",
            (400, 500),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.8,
            (200, 200, 200),
            3,
        )
        return mock

    def on_inspect_front_clicked(self):
        self.pending_inspection_action = "front"

        triggered = False
        if self.cam_thread and self.cam_thread.is_connected():
            triggered = self.cam_thread.send_software_trigger()

        if not triggered:
            # Fallback if camera is not connected or in simulation mode
            frame = self.get_current_frame()
            self.pending_inspection_action = None
            self.run_front_inspection(frame)

    def run_front_inspection(self, frame: np.ndarray):
        annotated, is_good, defects = self.cycle.process_front(frame)

        self.display_image(annotated)
        self.status_label.setText(
            f"Front Module Result: {'GOOD' if is_good else 'BAD'} | {', '.join(defects) if defects else 'No defects'}"
        )

        if not is_good:
            # STOP auto-transition timer and trigger NG Pop-up Modal IMMEDIATELY!
            self.front_auto_timer.stop()
            self.show_final_result_popup()
        else:
            self.update_button_states()
            # Start 30s auto-transition timer to Back Module if user doesn't press button
            self.front_auto_timer.start(int(self.config.front_to_back_timeout_sec * 1000))

    def on_front_auto_timeout(self):
        if self.cycle.step == InspectionStep.FRONT_DONE:
            print("[UI] Auto-transitioning from Front to Back Module after timeout...")
            self.on_inspect_back_clicked()

    def on_inspect_back_clicked(self):
        self.front_auto_timer.stop()
        self.pending_inspection_action = "back"

        triggered = False
        if self.cam_thread and self.cam_thread.is_connected():
            triggered = self.cam_thread.send_software_trigger()

        if not triggered:
            frame = self.get_current_frame()
            self.pending_inspection_action = None
            self.run_back_inspection(frame)

    def run_back_inspection(self, frame: np.ndarray):
        annotated, is_good, defects = self.cycle.process_back(frame)

        self.display_image(annotated)

        if not is_good:
            # STOP delay timer and trigger NG Pop-up Modal IMMEDIATELY!
            self.status_label.setText(
                f"Back Module Result: BAD | {', '.join(defects)}"
            )
            self.show_final_result_popup()
        else:
            self.status_label.setText(
                "Back Module Result: GOOD | Waiting 5s for Barcode..."
            )
            self.update_button_states()
            # Start 5000 ms delay timer before barcode polling
            self.barcode_delay_timer.start(self.config.barcode_initial_delay_ms)

    def start_barcode_polling(self):
        self.status_label.setText("Polling Barcode Model (soft-triggering every 1s)...")
        self.barcode_poll_timer.start(self.config.barcode_poll_interval_ms)

    def on_barcode_poll_tick(self):
        frame = self.get_current_frame()
        success, finished, barcode_str = self.cycle.process_barcode_attempt(frame)

        if self.cycle.barcode_warning_banner:
            self.warning_banner.setVisible(True)
        else:
            self.warning_banner.setVisible(False)

        if finished:
            self.barcode_poll_timer.stop()
            self.show_final_result_popup()

    def show_final_result_popup(self):
        self.total_triggers += 1
        if self.cycle.overall_good:
            self.good_count += 1
        else:
            self.bad_count += 1

        self.update_metrics()
        self.update_button_states()

        dlg = ResultDialog(
            is_good=self.cycle.overall_good,
            barcode_data=self.cycle.barcode_data,
            defects=self.cycle.all_defects,
            parent=self,
        )
        dlg.exec_()

        # After user clicks "Continue with Next Part", reset cycle for next part
        self.on_reset_cycle_clicked()

    def on_reset_cycle_clicked(self):
        self.front_auto_timer.stop()
        self.barcode_delay_timer.stop()
        self.barcode_poll_timer.stop()
        self.warning_banner.setVisible(False)

        self.cycle.reset_cycle()
        self.status_label.setText("Status: Reset complete. Ready for Front Module Inspection.")
        self.update_button_states()

    def update_metrics(self):
        self.good_value.setText(str(self.good_count))
        self.bad_value.setText(str(self.bad_count))
        rate = (self.bad_count / self.total_triggers * 100) if self.total_triggers > 0 else 0.0
        self.bad_percent_value.setText(f"{rate:.1f}%")

    def on_software_trigger(self):
        if self.cam_thread and self.cam_thread.is_connected():
            self.cam_thread.send_software_trigger()

        if self.cycle.step in (InspectionStep.FRONT_IDLE, InspectionStep.FRONT_DONE):
            self.on_inspect_front_clicked()
        elif self.cycle.step == InspectionStep.FRONT_DONE:
            self.on_inspect_back_clicked()

    def on_infer_local_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image for Inference", "", "Images (*.png *.jpg *.jpeg *.bmp)"
        )
        if not file_path:
            return

        img = cv2.imread(file_path)
        if img is None:
            QMessageBox.critical(self, "Error", f"Failed to load image: {file_path}")
            return

        self.last_frame = img.copy()

        if self.cycle.step in (InspectionStep.FRONT_IDLE, InspectionStep.FRONT_DONE):
            self.on_inspect_front_clicked()
        else:
            self.on_inspect_back_clicked()

    def on_mode_changed(self, idx):
        mode = self.mode_selector.itemData(idx)
        self.config.camera_trigger_mode = mode
        if self.cam_thread:
            self.cam_thread.set_trigger_mode(mode)

    def on_model_key_changed(self, key):
        if key == "scratch":
            cfg = self.config.scratch_config
        elif key == "cap":
            cfg = self.config.cap_config
        elif key == "defect_2":
            cfg = self.config.defect_2_config
        elif key == "part_presence":
            cfg = self.config.part_presence_config
        else:
            cfg = self.config.barcode_config

        self.conf_spin.setValue(cfg.confidence)
        self.imgsz_selector.setCurrentText(str(cfg.imgsz))

    def on_apply_model_config(self):
        key = self.model_key_selector.currentText()
        conf = float(self.conf_spin.value())
        imgsz = int(self.imgsz_selector.currentText())

        if key == "scratch":
            self.config.scratch_config.confidence = conf
            self.config.scratch_config.imgsz = imgsz
        elif key == "cap":
            self.config.cap_config.confidence = conf
            self.config.cap_config.imgsz = imgsz
        elif key == "defect_2":
            self.config.defect_2_config.confidence = conf
            self.config.defect_2_config.imgsz = imgsz
        elif key == "part_presence":
            self.config.part_presence_config.confidence = conf
            self.config.part_presence_config.imgsz = imgsz
        else:
            self.config.barcode_config.confidence = conf
            self.config.barcode_config.imgsz = imgsz

        self.detector.reload()
        QMessageBox.information(
            self, "Config Updated", f"Applied settings for model '{key}': Conf={conf}, ImgSz={imgsz}"
        )

    def on_apply_camera_settings(self):
        exp = float(self.exposure_spin.value())
        gain = float(self.gain_spin.value())
        if self.cam_thread:
            self.cam_thread.update_exposure(exp)
            self.cam_thread.update_gain(gain)
        QMessageBox.information(self, "Camera Config", f"Camera Exposure set to {exp}, Gain set to {gain}")

    def on_draw_roi_clicked(self, checked: bool):
        self.feed_label.set_drawing_mode(checked)

    def on_interactive_roi_selected(self, x, y, w, h):
        self.roi_x_spin.setValue(x)
        self.roi_y_spin.setValue(y)
        self.roi_w_spin.setValue(w)
        self.roi_h_spin.setValue(h)
        self.draw_roi_btn.setChecked(False)

    def closeEvent(self, event):
        self.front_auto_timer.stop()
        self.barcode_delay_timer.stop()
        self.barcode_poll_timer.stop()
        if self.cam_thread:
            self.cam_thread.stop()
        event.accept()


def run_app(config: AppConfig, detector=None) -> int:
    app = QApplication(sys.argv)
    window = MothersonUI(config, detector=detector)
    window.show()
    return app.exec_()
