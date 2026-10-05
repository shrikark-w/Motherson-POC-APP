import gc
import threading
import time
import traceback
from PyQt5.QtCore import QThread, pyqtSignal

import cv2
import numpy as np

from config import AppConfig


class BaumerCameraThread(QThread):
    frame_captured = pyqtSignal(object)
    camera_connected = pyqtSignal()
    camera_disconnected = pyqtSignal()

    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.camera = None
        self.cv_cap = None
        self._should_run = False
        self._connected = False
        self._camera_lock = threading.Lock()
        self._last_success_frame_time = 0.0
        self._frame_counter = 0

    def is_connected(self) -> bool:
        return self._connected

    def connect_camera(self) -> bool:
        # 1. Try NeoAPI Baumer Camera first
        try:
            import neoapi
            cam = neoapi.Cam()
            cam.Connect()
            if cam.IsConnected():
                try:
                    if self.config.camera_trigger_mode == "software":
                        cam.f.TriggerMode.value = neoapi.TriggerMode_On
                        cam.f.TriggerSource.value = neoapi.TriggerSource_Software
                    else:
                        cam.f.TriggerMode.value = neoapi.TriggerMode_Off

                    cam.f.ExposureTime.Set(self.config.baumer_exposure_time)
                    cam.f.Gain.Set(self.config.baumer_gain)
                except Exception as e:
                    print(f"[BaumerCam] Trigger/Params warning: {e}")

                self.camera = cam
                self._connected = True
                self.camera_connected.emit()
                print("[BaumerCam] NeoAPI Baumer Camera connected successfully!")
                return True
        except Exception as e:
            err_msg = str(e)
            if "NotConnectedException" not in err_msg and "No device" not in err_msg:
                print(f"[BaumerCam] NeoAPI connect note: {e}")

        # 2. Try OpenCV Camera (cv2.VideoCapture) fallback
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(0)

            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    self.cv_cap = cap
                    self._connected = True
                    self.camera_connected.emit()
                    print("[BaumerCam] OpenCV Industrial Camera connected on Device 0!")
                    return True
                else:
                    cap.release()
        except Exception as exc:
            print(f"[BaumerCam] OpenCV capture connect note: {exc}")

        self.camera = None
        self.cv_cap = None
        self._connected = False
        self.camera_disconnected.emit()
        return False

    def set_trigger_mode(self, mode: str):
        self.config.camera_trigger_mode = mode
        try:
            with self._camera_lock:
                if self.camera and self._connected:
                    import neoapi
                    if mode == "software":
                        self.camera.f.TriggerMode.value = neoapi.TriggerMode_On
                        self.camera.f.TriggerSource.value = neoapi.TriggerSource_Software
                    else:
                        self.camera.f.TriggerMode.value = neoapi.TriggerMode_Off
        except Exception as e:
            print(f"[BaumerCam] Failed to set trigger mode: {e}")

    def update_exposure(self, val: float):
        self.config.baumer_exposure_time = val
        try:
            with self._camera_lock:
                if self.camera and self._connected:
                    self.camera.f.ExposureTime.Set(val)
                    print(f"[BaumerCam] Exposure set to {val}")
        except Exception as e:
            print(f"[BaumerCam] Failed to set exposure: {e}")

    def update_gain(self, val: float):
        self.config.baumer_gain = val
        try:
            with self._camera_lock:
                if self.camera and self._connected:
                    self.camera.f.Gain.Set(val)
                    print(f"[BaumerCam] Gain set to {val}")
        except Exception as e:
            print(f"[BaumerCam] Failed to set gain: {e}")

    def send_software_trigger(self) -> bool:
        try:
            with self._camera_lock:
                if self.camera and getattr(self.camera, "IsConnected", lambda: False)():
                    self.camera.f.TriggerSoftware.Execute()
                    print("[BaumerCam] Software trigger executed via NeoAPI!")
                    return True
                elif self.cv_cap and self.cv_cap.isOpened():
                    ret, frame = self.cv_cap.read()
                    if ret and frame is not None:
                        self.frame_captured.emit(frame.copy())
                        print("[BaumerCam] Software trigger executed via OpenCV!")
                        return True
        except Exception as e:
            print(f"[BaumerCam] Software trigger error: {e}")
        return False

    def run(self):
        self._should_run = True
        self._last_success_frame_time = time.time()
        self._frame_counter = 0

        while self._should_run:
            if not self._connected or (not self.camera and not self.cv_cap):
                if not self.connect_camera():
                    for _ in range(10):
                        if not self._should_run:
                            break
                        time.sleep(0.2)
                    continue

                if self.camera:
                    try:
                        import neoapi
                        if self.config.camera_trigger_mode == "continuous":
                            self.camera.f.TriggerMode.value = neoapi.TriggerMode_Off
                        else:
                            self.camera.f.TriggerMode.value = neoapi.TriggerMode_On
                            self.camera.f.TriggerSource.value = neoapi.TriggerSource_Software
                        self.camera.StartStreaming()
                        print("[BaumerCam] NeoAPI Streaming started")
                    except Exception as e:
                        print(f"[BaumerCam] StartStreaming exception: {e}")

            # Grab frame from NeoAPI camera
            if self.camera and getattr(self.camera, "IsConnected", lambda: False)():
                image_data = None
                try:
                    image_data = self.camera.GetImage(int(self.config.baumer_get_image_timeout_ms))
                except Exception:
                    time.sleep(0.05)
                    continue

                if image_data is not None:
                    try:
                        frame = image_data.GetNPArray()
                        if frame is not None and frame.size > 0:
                            if frame.ndim == 2 or (frame.ndim == 3 and frame.shape[2] == 1):
                                try:
                                    pf = str(self.camera.f.PixelFormat.value)
                                    conv_map = {
                                        "BayerRG8": cv2.COLOR_BAYER_RG2BGR,
                                        "BayerBG8": cv2.COLOR_BAYER_BG2BGR,
                                        "BayerGR8": cv2.COLOR_BAYER_GR2BGR,
                                        "BayerGB8": cv2.COLOR_BAYER_GB2BGR,
                                    }
                                    code = conv_map.get(pf, cv2.COLOR_BAYER_BG2BGR)
                                    frame = cv2.cvtColor(frame, code)
                                except Exception:
                                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                            self.frame_captured.emit(frame.copy())
                            self._last_success_frame_time = time.time()
                            self._frame_counter += 1
                            if self._frame_counter % 50 == 0:
                                gc.collect()
                    finally:
                        del image_data

            # Grab frame from OpenCV camera fallback
            elif self.cv_cap and self.cv_cap.isOpened():
                ret, frame = self.cv_cap.read()
                if ret and frame is not None:
                    self.frame_captured.emit(frame.copy())
                    self._last_success_frame_time = time.time()
                    time.sleep(0.033)  # ~30 FPS preview
                else:
                    time.sleep(0.1)

            else:
                time.sleep(0.1)

        # Cleanup on thread exit
        try:
            if self.camera and getattr(self.camera, "IsConnected", lambda: False)():
                self.camera.StopStreaming()
                self.camera.Disconnect()
        except Exception:
            pass
        if self.cv_cap:
            try:
                self.cv_cap.release()
            except Exception:
                pass
        self.camera = None
        self.cv_cap = None
        self._connected = False

    def stop(self):
        self._should_run = False
        self.quit()
        self.wait(1500)
        self._connected = False
