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
from tkinter import ttk, messagebox, filedialog, font as tkfont

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
    BG = "#0F172A"
    CARD = "#111C32"
    CARD_2 = "#16223A"
    BORDER = "#24324D"
    TEXT = "#E5E7EB"
    MUTED = "#94A3B8"
    RED = "#FB7185"
    RED_DARK = "#9F1239"
    GREEN = "#34D399"
    GREEN_DARK = "#065F46"
    BLUE = "#60A5FA"
    AMBER = "#FBBF24"

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Pulsey Data Collector")
        self.root.geometry("1180x780")
        self.root.minsize(1060, 700)
        self.root.configure(bg=self.BG)

        self.state = CollectionState()
        self.event_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.client = None
        self.connected = False
        self.task_buttons: list[tk.Button] = []

        self.output_dir = tk.StringVar(value=str(Path("data")))
        self.subject = tk.StringVar(value="S01")
        self.broker = tk.StringVar(value=DEFAULT_BROKER)
        self.port = tk.IntVar(value=DEFAULT_PORT)
        self.username = tk.StringVar(value=DEFAULT_USERNAME)
        self.password = tk.StringVar(value=DEFAULT_PASSWORD)
        self.topic = tk.StringVar(value=TOPIC_FALL_RAW)
        self.status_text = tk.StringVar(value="Chưa kết nối MQTT")
        self.armed_text = tk.StringVar(value="Chưa chọn task")

        self._setup_style()
        self._build_ui()
        self.root.after(150, self._poll_events)

    def _setup_style(self) -> None:
        self.style = ttk.Style(self.root)
        try:
            self.style.theme_use("clam")
        except tk.TclError:
            pass

        self.default_font = tkfont.Font(root=self.root, family="Segoe UI", size=10)
        self.root.option_add("*Font", self.default_font)
        self.style.configure("Main.TFrame", background=self.BG)
        self.style.configure("Card.TFrame", background=self.CARD)
        self.style.configure("Soft.TFrame", background=self.CARD_2)
        self.style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT, font=("Segoe UI", 22, "bold"))
        self.style.configure("SubTitle.TLabel", background=self.BG, foreground=self.MUTED, font=("Segoe UI", 10))
        self.style.configure("CardTitle.TLabel", background=self.CARD, foreground=self.TEXT, font=("Segoe UI", 12, "bold"))
        self.style.configure("CardMuted.TLabel", background=self.CARD, foreground=self.MUTED, font=("Segoe UI", 9))
        self.style.configure("SoftTitle.TLabel", background=self.CARD_2, foreground=self.TEXT, font=("Segoe UI", 11, "bold"))
        self.style.configure("SoftMuted.TLabel", background=self.CARD_2, foreground=self.MUTED, font=("Segoe UI", 9))
        self.style.configure("TLabel", background=self.CARD, foreground=self.TEXT)
        self.style.configure(
            "TEntry",
            fieldbackground="#0B1220",
            background="#0B1220",
            foreground=self.TEXT,
            bordercolor=self.BORDER,
            lightcolor=self.BORDER,
            darkcolor=self.BORDER,
            padding=6,
        )
        self.style.map("TEntry", fieldbackground=[("focus", "#101A2E")])
        self.style.configure(
            "Primary.TButton",
            background="#2563EB",
            foreground="#FFFFFF",
            borderwidth=0,
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
        )
        self.style.map("Primary.TButton", background=[("active", "#1D4ED8")])
        self.style.configure(
            "Ghost.TButton",
            background=self.CARD_2,
            foreground=self.TEXT,
            borderwidth=0,
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
        )
        self.style.map("Ghost.TButton", background=[("active", "#1E293B")])
        self.style.configure(
            "Danger.TButton",
            background="#BE123C",
            foreground="#FFFFFF",
            borderwidth=0,
            padding=(14, 8),
            font=("Segoe UI", 10, "bold"),
        )
        self.style.map("Danger.TButton", background=[("active", "#9F1239")])

    def _build_ui(self) -> None:
        shell = ttk.Frame(self.root, style="Main.TFrame", padding=(20, 18, 20, 16))
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(2, weight=1)

        header = ttk.Frame(shell, style="Main.TFrame")
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Pulsey Fall Dataset Collector", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(
            header,
            text="Chọn task trước, thực hiện động tác, ESP32 gửi fall_raw, app tự lưu đúng một file CSV.",
            style="SubTitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        badge_row = ttk.Frame(header, style="Main.TFrame")
        badge_row.grid(row=0, column=1, rowspan=2, sticky="e")
        self.status_badge = tk.Label(
            badge_row,
            textvariable=self.status_text,
            bg="#334155",
            fg="#FFFFFF",
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
        )
        self.status_badge.pack(side="right", padx=(8, 0))
        self.armed_badge = tk.Label(
            badge_row,
            textvariable=self.armed_text,
            bg="#1E293B",
            fg=self.AMBER,
            font=("Segoe UI", 10, "bold"),
            padx=14,
            pady=8,
        )
        self.armed_badge.pack(side="right")

        config = self._card(shell)
        config.grid(row=1, column=0, sticky="ew", pady=(16, 14))
        config.columnconfigure(1, weight=3)
        config.columnconfigure(3, weight=1)
        config.columnconfigure(5, weight=1)

        ttk.Label(config, text="Cấu hình dữ liệu & MQTT", style="CardTitle.TLabel").grid(row=0, column=0, columnspan=6, sticky="w", pady=(0, 10))
        self._field(config, 1, 0, "Folder gốc", self.output_dir, width=42)
        ttk.Button(config, text="Chọn folder", style="Ghost.TButton", command=self.choose_output_dir).grid(row=1, column=2, sticky="ew", padx=(8, 16))
        self._field(config, 1, 3, "Subject", self.subject, width=10)

        self._field(config, 2, 0, "Broker", self.broker, width=30)
        self._field(config, 2, 3, "Port", self.port, width=10)
        self._field(config, 2, 5, "Topic", self.topic, width=24)

        self._field(config, 3, 0, "Username", self.username, width=30)
        self._field(config, 3, 3, "Password", self.password, width=22, show="*")
        mqtt_actions = ttk.Frame(config, style="Card.TFrame")
        mqtt_actions.grid(row=3, column=5, sticky="ew", padx=(10, 0))
        ttk.Button(mqtt_actions, text="Kết nối", style="Primary.TButton", command=self.connect_mqtt).pack(side="left", fill="x", expand=True)
        ttk.Button(mqtt_actions, text="Ngắt", style="Danger.TButton", command=self.disconnect_mqtt).pack(side="left", fill="x", expand=True, padx=(8, 0))

        content = ttk.Frame(shell, style="Main.TFrame")
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        fall_card = self._task_panel(content, "FALL", "label_binary = 1", self.RED, "fall")
        fall_card.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        not_fall_card = self._task_panel(content, "NOT FALL", "label_binary = 0", self.GREEN, "not_fall")
        not_fall_card.grid(row=0, column=1, sticky="nsew", padx=(8, 0))

        self._add_task_buttons(fall_card.body, FALL_TASKS, self.RED, self.RED_DARK)
        self._add_task_buttons(not_fall_card.body, NOT_FALL_TASKS, self.GREEN, self.GREEN_DARK)

        bottom = ttk.Frame(shell, style="Main.TFrame")
        bottom.grid(row=3, column=0, sticky="nsew", pady=(14, 0))
        bottom.columnconfigure(0, weight=0)
        bottom.columnconfigure(1, weight=1)

        actions = self._card(bottom)
        actions.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        ttk.Label(actions, text="Thao tác nhanh", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))
        ttk.Button(actions, text="Hủy task đang chờ", style="Danger.TButton", command=self.clear_task).pack(fill="x", pady=4)
        ttk.Button(actions, text="Mở folder data", style="Ghost.TButton", command=self.open_output_folder).pack(fill="x", pady=4)
        ttk.Button(actions, text="Tạo cây thư mục", style="Primary.TButton", command=self.create_folder_tree).pack(fill="x", pady=4)

        log_card = self._card(bottom)
        log_card.grid(row=0, column=1, sticky="nsew")
        log_card.rowconfigure(1, weight=1)
        log_card.columnconfigure(0, weight=1)
        ttk.Label(log_card, text="Log hệ thống", style="CardTitle.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))

        log_wrap = tk.Frame(log_card, bg="#0B1220", highlightbackground=self.BORDER, highlightthickness=1)
        log_wrap.grid(row=1, column=0, sticky="nsew")
        log_wrap.rowconfigure(0, weight=1)
        log_wrap.columnconfigure(0, weight=1)
        self.log_box = tk.Text(
            log_wrap,
            height=9,
            wrap="word",
            bg="#0B1220",
            fg="#D1D5DB",
            insertbackground=self.TEXT,
            relief="flat",
            bd=0,
            padx=12,
            pady=10,
            font=("Consolas", 10),
        )
        self.log_box.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_wrap, orient="vertical", command=self.log_box.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_box.configure(yscrollcommand=scrollbar.set)
        self.log_box.tag_configure("INFO", foreground="#D1D5DB")
        self.log_box.tag_configure("OK", foreground=self.GREEN)
        self.log_box.tag_configure("WARN", foreground=self.AMBER)
        self.log_box.tag_configure("ERR", foreground=self.RED)
        self.log_box.tag_configure("READY", foreground=self.BLUE)

        self.log("Cách dùng: chọn task trước → thực hiện động tác → ESP32 gửi sensor/fall_raw → app lưu đúng 1 file CSV.")
        self.log("Cây thư mục: data/fall/<task>/*.csv và data/not_fall/<task>/*.csv")

    def _card(self, parent: tk.Widget) -> ttk.Frame:
        frame = ttk.Frame(parent, style="Card.TFrame", padding=14)
        return frame

    def _field(self, parent: ttk.Frame, row: int, col: int, label: str, variable: tk.Variable, width: int, show: str | None = None) -> None:
        cell = ttk.Frame(parent, style="Card.TFrame")
        cell.grid(row=row, column=col, columnspan=2 if col == 0 else 1, sticky="ew", padx=(0, 10), pady=5)
        cell.columnconfigure(0, weight=1)
        ttk.Label(cell, text=label, style="CardMuted.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 4))
        entry = ttk.Entry(cell, textvariable=variable, width=width, show=show or "")
        entry.grid(row=1, column=0, sticky="ew")

    def _task_panel(self, parent: tk.Widget, title: str, subtitle: str, color: str, tag: str):
        outer = self._card(parent)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        top = ttk.Frame(outer, style="Card.TFrame")
        top.grid(row=0, column=0, sticky="ew", pady=(0, 12))
        top.columnconfigure(1, weight=1)
        dot = tk.Canvas(top, width=14, height=14, bg=self.CARD, highlightthickness=0)
        dot.grid(row=0, column=0, rowspan=2, sticky="w", padx=(0, 8))
        dot.create_oval(2, 2, 12, 12, fill=color, outline=color)
        ttk.Label(top, text=title, style="CardTitle.TLabel").grid(row=0, column=1, sticky="w")
        ttk.Label(top, text=subtitle, style="CardMuted.TLabel").grid(row=1, column=1, sticky="w")

        body = ttk.Frame(outer, style="Card.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=1)
        outer.body = body
        outer.tag = tag
        return outer

    def _add_task_buttons(self, parent: ttk.Frame, tasks: list[Task], color: str, active_color: str) -> None:
        for i, task in enumerate(tasks):
            btn = tk.Button(
                parent,
                text=f"{task.name}\n{task.display}",
                command=lambda t=task: self.arm_task(t),
                bg=self.CARD_2,
                fg=self.TEXT,
                activebackground=active_color,
                activeforeground="#FFFFFF",
                relief="flat",
                bd=0,
                padx=10,
                pady=12,
                cursor="hand2",
                justify="left",
                anchor="w",
                font=("Segoe UI", 10, "bold"),
                highlightthickness=1,
                highlightbackground=self.BORDER,
            )
            btn.default_bg = self.CARD_2
            btn.default_fg = self.TEXT
            btn.active_color = color
            btn.task = task
            btn.bind("<Enter>", lambda e, b=btn, c=color: b.configure(bg=c, fg="#FFFFFF"))
            btn.bind("<Leave>", lambda e, b=btn: self._refresh_task_button(b))
            btn.grid(row=i // 2, column=i % 2, sticky="nsew", padx=5, pady=5)
            self.task_buttons.append(btn)

        for col in range(2):
            parent.columnconfigure(col, weight=1)
        for row in range((len(tasks) + 1) // 2):
            parent.rowconfigure(row, weight=1)

    def _refresh_task_button(self, btn: tk.Button) -> None:
        current = self.state.current()
        if current is not None and current[0] == btn.task:
            btn.configure(bg=btn.active_color, fg="#FFFFFF")
        else:
            btn.configure(bg=btn.default_bg, fg=btn.default_fg)

    def _refresh_task_buttons(self) -> None:
        for btn in self.task_buttons:
            self._refresh_task_button(btn)

    def choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(title="Chọn folder gốc để lưu data")
        if selected:
            self.output_dir.set(selected)

    def create_folder_tree(self) -> None:
        root_dir = Path(self.output_dir.get())
        for task in ALL_TASKS:
            (root_dir / task.label / task.name).mkdir(parents=True, exist_ok=True)
        ensure_labels_file(root_dir)
        self.log(f"Đã tạo cây thư mục trong: {root_dir}", "OK")

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
        self.armed_badge.configure(bg="#422006", fg=self.AMBER)
        self._refresh_task_buttons()
        self.log(f"READY: {subject} | {task.label} | {task.name}. Bây giờ hãy thực hiện động tác.", "READY")

    def clear_task(self) -> None:
        self.state.disarm()
        self.armed_text.set("Chưa chọn task")
        self.armed_badge.configure(bg="#1E293B", fg=self.AMBER)
        self._refresh_task_buttons()
        self.log("Đã hủy task đang chờ.", "WARN")

    def connect_mqtt(self) -> None:
        if mqtt is None:
            messagebox.showerror("Thiếu thư viện", "Chưa cài paho-mqtt. Chạy: pip install paho-mqtt certifi")
            return
        if self.connected:
            self.log("MQTT đã kết nối rồi.", "OK")
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
            self.status_badge.configure(bg="#1D4ED8", fg="#FFFFFF")
            self.log(f"Đang kết nối MQTT broker={self.broker.get()} topic={self.topic.get()}", "READY")
        except Exception as exc:
            messagebox.showerror("MQTT error", str(exc))
            self.log(f"MQTT error: {exc}", "ERR")

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
        if hasattr(self, "status_badge"):
            self.status_badge.configure(bg="#334155", fg="#FFFFFF")
        self.log("Đã ngắt kết nối MQTT.", "WARN")

    def _on_connect(self, client, userdata, flags, reason_code, properties=None) -> None:
        if reason_code == 0:
            self.connected = True
            client.subscribe(self.topic.get(), qos=1)
            self.event_queue.put(("log_ok", f"MQTT connected. Subscribed: {self.topic.get()}"))
            self.event_queue.put(("status_ok", "MQTT đã kết nối"))
        else:
            self.event_queue.put(("log_err", f"MQTT connect failed: {reason_code}"))
            self.event_queue.put(("status_err", f"MQTT lỗi: {reason_code}"))

    def _on_disconnect(self, client, userdata, rc, properties=None) -> None:
        self.connected = False
        self.event_queue.put(("status_warn", "MQTT đã ngắt"))
        self.event_queue.put(("log_warn", f"MQTT disconnected rc={rc}"))

    def _on_message(self, client, userdata, msg) -> None:
        armed = self.state.consume()
        if armed is None:
            self.event_queue.put(("log_warn", "Nhận fall_raw nhưng chưa chọn task, bỏ qua. Hãy chọn task trước khi test."))
            return

        task, subject = armed
        decoded = decode_fall_raw_binary(msg.payload)
        if decoded is None:
            self.event_queue.put(("log_err", "Decode fall_raw thất bại. Chọn lại task và test lại."))
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
            self.event_queue.put(("log_err", f"Lỗi lưu CSV: {exc}"))

    def _poll_events(self) -> None:
        while True:
            try:
                kind, payload = self.event_queue.get_nowait()
            except queue.Empty:
                break

            if kind == "log":
                self.log(str(payload))
            elif kind == "log_ok":
                self.log(str(payload), "OK")
            elif kind == "log_warn":
                self.log(str(payload), "WARN")
            elif kind == "log_err":
                self.log(str(payload), "ERR")
            elif kind == "status":
                self.status_text.set(str(payload))
            elif kind == "status_ok":
                self.status_text.set(str(payload))
                self.status_badge.configure(bg=self.GREEN_DARK, fg="#FFFFFF")
            elif kind == "status_warn":
                self.status_text.set(str(payload))
                self.status_badge.configure(bg="#334155", fg="#FFFFFF")
            elif kind == "status_err":
                self.status_text.set(str(payload))
                self.status_badge.configure(bg=self.RED_DARK, fg="#FFFFFF")
            elif kind == "saved":
                info = payload
                task = info["task"]
                self.armed_text.set("Chưa chọn task")
                self.armed_badge.configure(bg="#1E293B", fg=self.AMBER)
                self._refresh_task_buttons()
                self.log(
                    f"SAVED: {info['path']} | subject={info['subject']} | "
                    f"label={task.label} | task={task.name} | samples={info['samples']} | pre={info['pre']}",
                    "OK",
                )
                self.log("Đã lưu đúng 1 window. Hãy chọn task tiếp theo rồi mới test tiếp.", "INFO")

        self.root.after(150, self._poll_events)

    def log(self, message: str, level: str = "INFO") -> None:
        now = time.strftime("%H:%M:%S")
        tag = level if level in {"INFO", "OK", "WARN", "ERR", "READY"} else "INFO"
        self.log_box.insert("end", f"[{now}] {message}\n", tag)
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
