from __future__ import annotations

import csv
import queue
import re
import ssl
import struct
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

try:
    import certifi
except ModuleNotFoundError:
    certifi = None

try:
    import paho.mqtt.client as mqtt
except ModuleNotFoundError:
    mqtt = None

try:
    from app.core.config import settings
except Exception:
    settings = None

TOPIC_FALL_RAW = "sensor/fall_raw"
SAMPLE_INTERVAL_S = 0.01

DEFAULT_BROKER = getattr(settings, "mqtt_broker", "") if settings else ""
DEFAULT_PORT = int(getattr(settings, "mqtt_port", 8883) if settings else 8883)
DEFAULT_USERNAME = getattr(settings, "mqtt_username", "") if settings else ""
DEFAULT_PASSWORD = getattr(settings, "mqtt_password", "") if settings else ""


@dataclass(frozen=True)
class Task:
    label: str
    name: str
    display: str


FALL_TASKS = [
    Task("fall", "forward_fall", "Té về trước"),
    Task("fall", "backward_fall", "Té về sau"),
    Task("fall", "left_fall", "Té sang trái"),
    Task("fall", "right_fall", "Té sang phải"),
    Task("fall", "sit_fall", "Té khi ngồi / té khỏi ghế"),
]

NOT_FALL_TASKS = [
    Task("not_fall", "walk", "Đi bộ"),
    Task("not_fall", "run", "Chạy nhẹ / bước nhanh"),
    Task("not_fall", "sit_down", "Ngồi xuống"),
    Task("not_fall", "stand_up", "Đứng dậy"),
    Task("not_fall", "stairs", "Lên/xuống cầu thang"),
    Task("not_fall", "jump", "Nhảy nhẹ"),
    Task("not_fall", "bend", "Cúi người nhặt đồ"),
    Task("not_fall", "arm_swing", "Vung tay / xoay người"),
    Task("not_fall", "device_bump", "Va chạm nhẹ thiết bị"),
    Task("not_fall", "lying", "Nằm"),
    Task("not_fall", "other", "Không té khác"),
]

ALL_TASKS = FALL_TASKS + NOT_FALL_TASKS


class CollectionState:
    def __init__(self) -> None:
        self._lock = Lock()
        self.armed_task: Task | None = None
        self.subject = "S01"

    def arm(self, task: Task, subject: str) -> None:
        with self._lock:
            self.armed_task = task
            self.subject = normalize_subject(subject)

    def disarm(self) -> None:
        with self._lock:
            self.armed_task = None

    def consume(self) -> tuple[Task, str] | None:
        with self._lock:
            if self.armed_task is None:
                return None
            task = self.armed_task
            subject = self.subject
            self.armed_task = None
            return task, subject

    def current(self) -> tuple[Task, str] | None:
        with self._lock:
            if self.armed_task is None:
                return None
            return self.armed_task, self.subject


def normalize_subject(raw: str) -> str:
    value = str(raw or "S01").strip().upper()
    if value.startswith("SA"):
        value = "S" + value[2:]
    if value.startswith("S"):
        digits = re.sub(r"\D", "", value[1:])
    else:
        digits = re.sub(r"\D", "", value)
    if not digits:
        digits = "1"
    return f"S{int(digits):02d}"


def decode_fall_raw_binary(buf: bytes) -> dict | None:
    try:
        if len(buf) < 8:
            return None

        offset = 0
        trigger_ts = struct.unpack_from(">I", buf, offset)[0]
        offset += 4
        num_samples = struct.unpack_from(">H", buf, offset)[0]
        offset += 2
        pre_samples = buf[offset]
        offset += 1
        reason_len = buf[offset]
        offset += 1

        if offset + reason_len > len(buf):
            return None

        reason = buf[offset: offset + reason_len].decode("utf-8", errors="replace")
        offset += reason_len

        samples = []
        for _ in range(num_samples):
            if offset + 16 > len(buf):
                return None
            raw = struct.unpack_from(">8h", buf, offset)
            offset += 16
            samples.append({
                "ax": raw[0] / 1000.0,
                "ay": raw[1] / 1000.0,
                "az": raw[2] / 1000.0,
                "gx": raw[3] / 10.0,
                "gy": raw[4] / 10.0,
                "gz": raw[5] / 10.0,
                "acc_mag": raw[6] / 1000.0,
                "jerk": raw[7] / 1000.0,
            })

        return {
            "trigger_ts": trigger_ts,
            "num_samples": num_samples,
            "pre_samples": pre_samples,
            "reason": reason,
            "samples": samples,
        }
    except Exception:
        traceback.print_exc()
        return None


def next_file_path(root_dir: Path, subject: str, task: Task) -> tuple[Path, int]:
    task_dir = root_dir / task.label / task.name
    task_dir.mkdir(parents=True, exist_ok=True)

    pattern = f"{subject}_{task.name}_*.csv"
    max_repeat = 0
    for path in task_dir.glob(pattern):
        match = re.search(r"_(\d+)\.csv$", path.name)
        if match:
            max_repeat = max(max_repeat, int(match.group(1)))

    repeat = max_repeat + 1
    while True:
        file_name = f"{subject}_{task.name}_{repeat:03d}.csv"
        csv_path = task_dir / file_name
        if not csv_path.exists():
            return csv_path, repeat
        repeat += 1


def ensure_labels_file(root_dir: Path) -> Path:
    labels_path = root_dir / "labels.csv"
    root_dir.mkdir(parents=True, exist_ok=True)
    if labels_path.exists():
        return labels_path
    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "subject", "label", "label_binary", "task_name", "repeat",
            "received_at", "trigger_ts", "num_samples", "pre_samples", "reason",
        ])
        writer.writeheader()
    return labels_path


def append_label_row(root_dir: Path, row: dict) -> None:
    labels_path = ensure_labels_file(root_dir)
    with labels_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "file", "subject", "label", "label_binary", "task_name", "repeat",
            "received_at", "trigger_ts", "num_samples", "pre_samples", "reason",
        ])
        writer.writerow(row)


def save_window(root_dir: Path, subject: str, task: Task, decoded: dict) -> Path:
    csv_path, repeat = next_file_path(root_dir, subject, task)
    pre_samples = int(decoded.get("pre_samples") or 0)
    samples = decoded.get("samples") or []
    label_binary = 1 if task.label == "fall" else 0

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "trial_id", "label", "label_binary", "task_name",
            "TimeStamp(s)", "FrameCounter", "SampleIndex", "RelativeIndex",
            "AccX", "AccY", "AccZ", "GyrX", "GyrY", "GyrZ",
            "AccMag", "Jerk", "PreTrigger", "TriggerIndex", "Reason",
        ])
        writer.writeheader()
        trial_id = csv_path.stem
        for idx, sample in enumerate(samples):
            writer.writerow({
                "trial_id": trial_id,
                "label": task.label,
                "label_binary": label_binary,
                "task_name": task.name,
                "TimeStamp(s)": f"{idx * SAMPLE_INTERVAL_S:.2f}",
                "FrameCounter": idx + 1,
                "SampleIndex": idx,
                "RelativeIndex": idx - pre_samples,
                "AccX": sample["ax"],
                "AccY": sample["ay"],
                "AccZ": sample["az"],
                "GyrX": sample["gx"],
                "GyrY": sample["gy"],
                "GyrZ": sample["gz"],
                "AccMag": sample["acc_mag"],
                "Jerk": sample["jerk"],
                "PreTrigger": idx < pre_samples,
                "TriggerIndex": pre_samples,
                "Reason": decoded.get("reason"),
            })

    rel_path = csv_path.relative_to(root_dir)
    append_label_row(root_dir, {
        "file": str(rel_path).replace("\\", "/"),
        "subject": subject,
        "label": task.label,
        "label_binary": label_binary,
        "task_name": task.name,
        "repeat": repeat,
        "received_at": int(time.time()),
        "trigger_ts": decoded.get("trigger_ts"),
        "num_samples": decoded.get("num_samples"),
        "pre_samples": decoded.get("pre_samples"),
        "reason": decoded.get("reason"),
    })
    return csv_path


class CollectorApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fall / Not Fall Data Collector")
        self.root.geometry("980x720")

        self.state = CollectionState()
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.client = None
        self.connected = False

        self.output_dir = tk.StringVar(value=str(Path("data")))
        self.subject = tk.StringVar(value="S01")
        self.broker = tk.StringVar(value=DEFAULT_BROKER)
        self.port = tk.IntVar(value=DEFAULT_PORT)
        self.username = tk.StringVar(value=DEFAULT_USERNAME)
        self.password = tk.StringVar(value=DEFAULT_PASSWORD)
        self.topic = tk.StringVar(value=TOPIC_FALL_RAW)
        self.status_text = tk.StringVar(value="Chưa kết nối MQTT")
        self.armed_text = tk.StringVar(value="Chưa chọn task")

        self._build_ui()
        self.root.after(150, self._poll_events)

    def _build_ui(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)

        config = ttk.LabelFrame(main, text="Cấu hình lưu dữ liệu và MQTT", padding=10)
        config.pack(fill="x")

        ttk.Label(config, text="Folder gốc:").grid(row=0, column=0, sticky="w")
        ttk.Entry(config, textvariable=self.output_dir, width=55).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(config, text="Chọn...", command=self.choose_output_dir).grid(row=0, column=2, padx=4)

        ttk.Label(config, text="Subject:").grid(row=0, column=3, sticky="w", padx=(20, 0))
        ttk.Entry(config, textvariable=self.subject, width=10).grid(row=0, column=4, sticky="w", padx=6)

        ttk.Label(config, text="Broker:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config, textvariable=self.broker, width=35).grid(row=1, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Label(config, text="Port:").grid(row=1, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(config, textvariable=self.port, width=8).grid(row=1, column=3, sticky="w", padx=6, pady=(8, 0))

        ttk.Label(config, text="Username:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config, textvariable=self.username, width=35).grid(row=2, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Label(config, text="Password:").grid(row=2, column=2, sticky="e", pady=(8, 0))
        ttk.Entry(config, textvariable=self.password, show="*", width=28).grid(row=2, column=3, columnspan=2, sticky="w", padx=6, pady=(8, 0))

        ttk.Label(config, text="Topic:").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(config, textvariable=self.topic, width=35).grid(row=3, column=1, sticky="ew", padx=6, pady=(8, 0))
        ttk.Button(config, text="Kết nối MQTT", command=self.connect_mqtt).grid(row=3, column=2, padx=4, pady=(8, 0))
        ttk.Button(config, text="Ngắt kết nối", command=self.disconnect_mqtt).grid(row=3, column=3, padx=4, pady=(8, 0))

        config.columnconfigure(1, weight=1)

        status = ttk.Frame(main)
        status.pack(fill="x", pady=10)
        ttk.Label(status, textvariable=self.status_text, font=("Segoe UI", 10, "bold")).pack(side="left")
        ttk.Label(status, textvariable=self.armed_text, font=("Segoe UI", 10, "bold")).pack(side="right")

        body = ttk.Frame(main)
        body.pack(fill="both", expand=True)

        fall_frame = ttk.LabelFrame(body, text="FALL - label_binary = 1", padding=10)
        fall_frame.pack(side="left", fill="both", expand=True, padx=(0, 6))

        not_fall_frame = ttk.LabelFrame(body, text="NOT FALL - label_binary = 0", padding=10)
        not_fall_frame.pack(side="left", fill="both", expand=True, padx=(6, 0))

        self._add_task_buttons(fall_frame, FALL_TASKS)
        self._add_task_buttons(not_fall_frame, NOT_FALL_TASKS)

        actions = ttk.Frame(main)
        actions.pack(fill="x", pady=8)
        ttk.Button(actions, text="Hủy task đang chờ", command=self.clear_task).pack(side="left")
        ttk.Button(actions, text="Mở folder data", command=self.open_output_folder).pack(side="left", padx=8)
        ttk.Button(actions, text="Tạo sẵn cây thư mục", command=self.create_folder_tree).pack(side="left")

        log_frame = ttk.LabelFrame(main, text="Log", padding=8)
        log_frame.pack(fill="both", expand=True)
        self.log_box = tk.Text(log_frame, height=12, wrap="word")
        self.log_box.pack(fill="both", expand=True)

        self.log("Cách dùng: chọn task trước → thực hiện động tác → ESP32 gửi sensor/fall_raw → app lưu đúng 1 file CSV.")
        self.log("Cây thư mục: data/fall/<task>/*.csv và data/not_fall/<task>/*.csv")

    def _add_task_buttons(self, parent: ttk.Frame, tasks: list[Task]) -> None:
        for i, task in enumerate(tasks):
            text = f"{task.name}\n{task.display}"
            btn = ttk.Button(parent, text=text, command=lambda t=task: self.arm_task(t))
            btn.grid(row=i // 2, column=i % 2, sticky="ew", padx=5, pady=5)
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)

    def choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="Chọn folder gốc để lưu data")
        if selected:
            self.output_dir.set(selected)

    def create_folder_tree(self) -> None:
        root_dir = Path(self.output_dir.get())
        for task in ALL_TASKS:
            (root_dir / task.label / task.name).mkdir(parents=True, exist_ok=True)
        ensure_labels_file(root_dir)
        self.log(f"Đã tạo cây thư mục trong: {root_dir}")

    def open_output_folder(self) -> None:
        root_dir = Path(self.output_dir.get())
        root_dir.mkdir(parents=True, exist_ok=True)
        if sys.platform.startswith("win"):
            import os
            os.startfile(root_dir)
        elif sys.platform == "darwin":
            import subprocess
            subprocess.Popen(["open", str(root_dir)])
        else:
            import subprocess
            subprocess.Popen(["xdg-open", str(root_dir)])

    def arm_task(self, task: Task) -> None:
        subject = normalize_subject(self.subject.get())
        self.subject.set(subject)
        self.state.arm(task, subject)
        self.armed_text.set(f"Đang chờ: {subject} | {task.label} | {task.name}")
        self.log(f"READY: {subject} | {task.label} | {task.name}. Bây giờ hãy thực hiện động tác.")

    def clear_task(self) -> None:
        self.state.disarm()
        self.armed_text.set("Chưa chọn task")
        self.log("Đã hủy task đang chờ.")

    def connect_mqtt(self) -> None:
        if mqtt is None:
            messagebox.showerror("Thiếu thư viện", "Chưa cài paho-mqtt. Chạy: pip install paho-mqtt certifi")
            return
        if self.connected:
            self.log("MQTT đã kết nối rồi.")
            return
        try:
            client_id = f"fall_data_ui_{int(time.time())}"
            client = mqtt.Client(callback_api_version=mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
            client.username_pw_set(self.username.get(), self.password.get())
            if certifi is not None:
                client.tls_set(ca_certs=certifi.where(), tls_version=ssl.PROTOCOL_TLS_CLIENT)
            else:
                client.tls_set(tls_version=ssl.PROTOCOL_TLS_CLIENT)

            client.on_connect = self._on_connect
            client.on_disconnect = self._on_disconnect
            client.on_message = self._on_message
            client.connect_async(self.broker.get(), int(self.port.get()))
            client.loop_start()
            self.client = client
            self.status_text.set("Đang kết nối MQTT...")
            self.log(f"Đang kết nối MQTT broker={self.broker.get()} topic={self.topic.get()}")
        except Exception as exc:
            messagebox.showerror("MQTT error", str(exc))
            self.log(f"MQTT error: {exc}")

    def disconnect_mqtt(self) -> None:
        if self.client is not None:
            try:
                self.client.loop_stop()
                self.client.disconnect()
            except Exception:
                pass
        self.client = None
        self.connected = False
        self.status_text.set("Đã ngắt MQTT")
        self.log("Đã ngắt kết nối MQTT.")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if reason_code == 0:
            self.connected = True
            client.subscribe(self.topic.get(), qos=1)
            self.event_queue.put(("log", f"MQTT connected. Subscribed: {self.topic.get()}"))
            self.event_queue.put(("status", "MQTT đã kết nối"))
        else:
            self.event_queue.put(("log", f"MQTT connect failed: {reason_code}"))
            self.event_queue.put(("status", f"MQTT lỗi: {reason_code}"))

    def _on_disconnect(self, client, userdata, rc, properties=None) -> None:
        self.connected = False
        self.event_queue.put(("status", "MQTT đã ngắt"))
        self.event_queue.put(("log", f"MQTT disconnected rc={rc}"))

    def _on_message(self, client, userdata, msg) -> None:
        armed = self.state.consume()
        if armed is None:
            self.event_queue.put(("log", "Nhận fall_raw nhưng chưa chọn task, bỏ qua. Hãy chọn task trước khi test."))
            return

        task, subject = armed
        decoded = decode_fall_raw_binary(msg.payload)
        if decoded is None:
            self.event_queue.put(("log", "Decode fall_raw thất bại. Chọn lại task và test lại."))
            return

        try:
            root_dir = Path(self.output_dir.get())
            saved_path = save_window(root_dir, subject, task, decoded)
            self.event_queue.put(("saved", {
                "path": str(saved_path),
                "task": task,
                "subject": subject,
                "samples": decoded.get("num_samples"),
                "pre": decoded.get("pre_samples"),
            }))
        except Exception as exc:
            self.event_queue.put(("log", f"Lỗi lưu CSV: {exc}"))

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self.log(str(payload))
            elif kind == "status":
                self.status_text.set(str(payload))
            elif kind == "saved":
                info = payload
                task = info["task"]
                self.armed_text.set("Chưa chọn task")
                self.log(
                    f"SAVED: {info['path']} | subject={info['subject']} | "
                    f"label={task.label} | task={task.name} | samples={info['samples']} | pre={info['pre']}"
                )
                self.log("Đã lưu đúng 1 window. Hãy chọn task tiếp theo rồi mới test tiếp.")

        self.root.after(150, self._poll_events)

    def log(self, message: str) -> None:
        now = time.strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{now}] {message}\n")
        self.log_box.see("end")

    def on_close(self) -> None:
        self.disconnect_mqtt()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = CollectorApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
