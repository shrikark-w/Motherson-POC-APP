import sys
import traceback
from datetime import datetime
from pathlib import Path


class StreamTee:
    """Tees stdout and stderr to both console and app_log.txt."""

    def __init__(self, original_stream, filepath):
        self.original_stream = original_stream
        self.filepath = filepath

    def write(self, data):
        self.original_stream.write(data)
        self.original_stream.flush()
        if data:
            try:
                with open(self.filepath, "a", encoding="utf-8") as f:
                    f.write(data)
            except Exception:
                pass

    def flush(self):
        self.original_stream.flush()


# Redirect standard output and error to app_log.txt
log_path = "app_log.txt"
try:
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(
            f"\n--- MOTHERSON App Startup at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---\n"
        )
except Exception:
    pass

sys.stdout = StreamTee(sys.stdout, log_path)
sys.stderr = StreamTee(sys.stderr, log_path)


def exception_hook(exctype, value, tb):
    tb_str = "".join(traceback.format_exception(exctype, value, tb))
    sys.stderr.write(f"CRITICAL: Uncaught exception occurred:\n{tb_str}\n")
    sys.__excepthook__(exctype, value, tb)


sys.excepthook = exception_hook


from config import AppConfig
from vision_engine import VisionEngine


def main() -> int:
    config = AppConfig(
        company_name="MOTHERSON",
        app_name="Automotive Visual Inspection Platform",
    )

    # Initialize PyTorch and YOLO models before launching PyQt GUI thread
    print("[Main] Initializing Motherson Vision Engine...")
    detector = VisionEngine(config)

    from ui import run_app

    print("[Main] Launching Motherson Quality Dashboard UI...")
    return run_app(config, detector=detector)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        tb_str = traceback.format_exc()
        sys.stderr.write(f"CRITICAL: Application startup failed:\n{tb_str}\n")
