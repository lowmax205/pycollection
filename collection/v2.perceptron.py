"""
Combined (Single + MLP) digit recognition app using ttkbootstrap.

UI library: ttkbootstrap (themed tkinter)
Model toggle:
- Single: single-layer perceptron (sigmoid)
- MLP: 1 hidden layer (sigmoid)

Dataset format matches the original scripts (JSON inside .txt).

Dependencies:
  pip install numpy ttkbootstrap

Run:
  python perceptron.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

try:
    import ttkbootstrap as tb
    from ttkbootstrap.constants import BOTH, DISABLED, END, LEFT, NORMAL, RIGHT, TOP, X, Y
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "ttkbootstrap is required. Install with: pip install ttkbootstrap"
    ) from exc

import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText


GRID_ROWS = 7
GRID_COLS = 5
CELL_SIZE = 34
DEFAULT_DATASET_FILE = "digits_dataset.txt"


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _sigmoid_derivative(sigmoid_out: np.ndarray) -> np.ndarray:
    return sigmoid_out * (1.0 - sigmoid_out)


@dataclass
class Dataset:
    grid_rows: int = GRID_ROWS
    grid_cols: int = GRID_COLS
    samples: List[List[float]] = None
    labels: List[int] = None
    label_mapping: Dict[str, int] = None
    reverse_label_mapping: Dict[int, str] = None

    def __post_init__(self) -> None:
        self.samples = self.samples or []
        self.labels = self.labels or []
        self.label_mapping = self.label_mapping or {}
        self.reverse_label_mapping = self.reverse_label_mapping or {}

    @property
    def input_size(self) -> int:
        return self.grid_rows * self.grid_cols

    def add(self, letter: str, flat_pixels: np.ndarray) -> None:
        letter = letter.strip()
        if len(letter) != 1 or not letter.isdigit():
            raise ValueError("Label must be a single digit 0-9")
        if flat_pixels.shape != (self.input_size,):
            raise ValueError(f"Expected flat pixel vector of shape ({self.input_size},)")
        if int(np.sum(flat_pixels)) == 0:
            raise ValueError("Empty drawing")

        if letter not in self.label_mapping:
            idx = len(self.label_mapping)
            self.label_mapping[letter] = idx
            self.reverse_label_mapping[idx] = letter

        self.samples.append(flat_pixels.astype(float).tolist())
        self.labels.append(self.label_mapping[letter])

    def to_json_obj(self) -> dict:
        return {
            "samples": self.samples,
            "labels": self.labels,
            "label_mapping": self.label_mapping,
            "reverse_label_mapping": {str(k): v for k, v in self.reverse_label_mapping.items()},
            "grid_rows": self.grid_rows,
            "grid_cols": self.grid_cols,
        }

    @staticmethod
    def from_json_obj(obj: dict) -> "Dataset":
        ds = Dataset(
            grid_rows=int(obj.get("grid_rows", GRID_ROWS)),
            grid_cols=int(obj.get("grid_cols", GRID_COLS)),
        )
        ds.samples = [list(map(float, s)) for s in obj.get("samples", [])]
        ds.labels = [int(x) for x in obj.get("labels", [])]
        ds.label_mapping = {str(k): int(v) for k, v in obj.get("label_mapping", {}).items()}
        rlm = obj.get("reverse_label_mapping", {})
        ds.reverse_label_mapping = {int(k): str(v) for k, v in rlm.items()}
        return ds

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_json_obj(), f, indent=2)

    @staticmethod
    def load(path: str) -> "Dataset":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return Dataset.from_json_obj(obj)

    def stats_text(self) -> str:
        if not self.samples:
            return "No data loaded"

        counts: Dict[str, int] = {}
        for label_idx in self.labels:
            letter = self.reverse_label_mapping[int(label_idx)]
            counts[letter] = counts.get(letter, 0) + 1

        lines = [
            f"Total samples: {len(self.samples)}",
            f"Unique digits: {len(self.label_mapping)}",
            "",
            "Digit distribution:",
        ]
        for letter in sorted(counts.keys()):
            lines.append(f"  {letter}: {counts[letter]} samples")
        return "\n".join(lines)


class SinglePerceptron:
    def __init__(self, input_size: int, output_size: int, learning_rate: float) -> None:
        self.weights = np.random.randn(input_size, output_size) * 0.5
        self.bias = np.zeros((1, output_size))
        self.learning_rate = float(learning_rate)

    def forward(self, X: np.ndarray) -> np.ndarray:
        return _sigmoid(np.dot(X, self.weights) + self.bias)

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        out = self.forward(X)
        error = y - out
        delta = error * _sigmoid_derivative(out)
        self.weights += X.T.dot(delta) * self.learning_rate
        self.bias += np.sum(delta, axis=0, keepdims=True) * self.learning_rate
        return float(np.mean((y - out) ** 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


class MultiLayerPerceptron:
    def __init__(self, input_size: int, hidden_size: int, output_size: int, learning_rate: float) -> None:
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.5
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.5
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        self.learning_rate = float(learning_rate)

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = _sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = _sigmoid(self.output_input)
        return self.output

    def train_step(self, X: np.ndarray, y: np.ndarray) -> float:
        out = self.forward(X)
        output_error = y - out
        output_delta = output_error * _sigmoid_derivative(out)

        hidden_error = output_delta.dot(self.weights_hidden_output.T)
        hidden_delta = hidden_error * _sigmoid_derivative(self.hidden_output)

        self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * self.learning_rate
        self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * self.learning_rate

        self.weights_input_hidden += X.T.dot(hidden_delta) * self.learning_rate
        self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * self.learning_rate

        return float(np.mean((y - out) ** 2))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)


class PixelGrid:
    def __init__(self, rows: int, cols: int) -> None:
        self.rows = rows
        self.cols = cols
        self.clear()

    def clear(self) -> None:
        self.pixels = [[0 for _ in range(self.cols)] for _ in range(self.rows)]

    def set(self, r: int, c: int, v: int) -> None:
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.pixels[r][c] = 1 if v else 0

    def to_flat(self) -> np.ndarray:
        return np.array([p for row in self.pixels for p in row], dtype=float)


class App:
    def __init__(self) -> None:
        self.root = tb.Window(themename="flatly")
        self.root.title("Digit Recognition")
        self.root.geometry("1150x680")
        self.root.minsize(980, 600)

        self.dataset = Dataset()
        self.grid = PixelGrid(GRID_ROWS, GRID_COLS)

        self.model_single: Optional[SinglePerceptron] = None
        self.model_mlp: Optional[MultiLayerPerceptron] = None

        self._dragging = False
        self._rect_ids: List[List[int]] = [[0 for _ in range(GRID_COLS)] for _ in range(GRID_ROWS)]

        self._build_ui()
        self._load_default_dataset_if_present()
        self._refresh_dataset_info()
        self._refresh_model_controls()

    # ---------- UI ----------

    def _build_ui(self) -> None:
        main = tb.Frame(self.root, padding=10)
        main.pack(fill=BOTH, expand=True)

        left = tb.Frame(main)
        left.pack(side=LEFT, fill=Y, padx=(0, 10))

        right = tb.Frame(main)
        right.pack(side=RIGHT, fill=BOTH, expand=True)

        tb.Label(left, text="Draw", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        tb.Label(left, text="Click/drag to paint digits (5x7)").pack(anchor="w", pady=(0, 6))

        self.canvas = tk.Canvas(left, width=GRID_COLS * CELL_SIZE, height=GRID_ROWS * CELL_SIZE, bg="white", highlightthickness=1)
        self.canvas.pack(pady=(0, 8))

        self.canvas.bind("<ButtonPress-1>", self._on_down)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_up)

        btn_row = tb.Frame(left)
        btn_row.pack(fill=X)
        tb.Button(btn_row, text="Clear", command=self.clear_canvas, bootstyle="secondary").pack(side=LEFT, fill=X, expand=True, padx=(0, 6))
        tb.Button(btn_row, text="Recognize", command=self.recognize, bootstyle="success").pack(side=LEFT, fill=X, expand=True)

        add_frame = tb.Labelframe(left, text="Add sample", padding=10)
        add_frame.pack(fill=X, pady=(10, 0))
        self.label_entry = tb.Entry(add_frame)
        self.label_entry.pack(fill=X)
        tb.Button(add_frame, text="Add to dataset", command=self.add_sample, bootstyle="warning").pack(fill=X, pady=(6, 0))

        result_frame = tb.Labelframe(left, text="Result", padding=10)
        result_frame.pack(fill=X, pady=(10, 0))
        self.result_var = tk.StringVar(value="—")
        self.conf_var = tk.StringVar(value="")
        tb.Label(result_frame, textvariable=self.result_var, font=("Segoe UI", 28, "bold")).pack(anchor="center")
        tb.Label(result_frame, textvariable=self.conf_var).pack(anchor="center")
        self.top3_text = tb.Text(result_frame, height=4, width=26)
        self.top3_text.pack(fill=X, pady=(6, 0))
        self.top3_text.configure(state=DISABLED)

        # Right side: one-screen layout (no tabs)
        train_tab = tb.Labelframe(right, text="Training", padding=12)
        train_tab.pack(fill=X)

        data_tab = tb.Labelframe(right, text="Dataset", padding=12)
        data_tab.pack(fill=BOTH, expand=True, pady=(10, 0))

        log_tab = tb.Labelframe(right, text="Log", padding=12)
        log_tab.pack(fill=BOTH, expand=True, pady=(10, 0))

        # Training
        tb.Label(train_tab, text="Model & parameters", font=("Segoe UI", 12, "bold")).pack(anchor="w")

        self.model_var = tk.StringVar(value="Single")
        model_row = tb.Frame(train_tab)
        model_row.pack(fill=X, pady=(10, 0))
        tb.Label(model_row, text="Model", width=12).pack(side=LEFT)
        self.model_combo = tb.Combobox(model_row, textvariable=self.model_var, values=["Single", "MLP"], state="readonly")
        self.model_combo.pack(side=LEFT, fill=X, expand=True)
        self.model_combo.bind("<<ComboboxSelected>>", lambda _e: self._refresh_model_controls())

        self.hidden_row = tb.Frame(train_tab)
        self.hidden_row.pack(fill=X, pady=(10, 0))
        tb.Label(self.hidden_row, text="Hidden", width=12).pack(side=LEFT)
        self.hidden_var = tk.IntVar(value=10)
        self.hidden_scale = tb.Scale(self.hidden_row, from_=5, to=30, variable=self.hidden_var)
        self.hidden_scale.pack(side=LEFT, fill=X, expand=True)
        self.hidden_lbl = tb.Label(self.hidden_row, text="10")
        self.hidden_lbl.pack(side=LEFT, padx=(8, 0))
        self.hidden_var.trace_add("write", lambda *_: self.hidden_lbl.configure(text=str(self.hidden_var.get())))

        lr_row = tb.Frame(train_tab)
        lr_row.pack(fill=X, pady=(10, 0))
        tb.Label(lr_row, text="Learning rate", width=12).pack(side=LEFT)
        self.lr_var = tk.DoubleVar(value=0.10)
        self.lr_scale = tb.Scale(lr_row, from_=0.01, to=1.0, variable=self.lr_var)
        self.lr_scale.pack(side=LEFT, fill=X, expand=True)
        self.lr_lbl = tb.Label(lr_row, text="0.10")
        self.lr_lbl.pack(side=LEFT, padx=(8, 0))
        self.lr_var.trace_add("write", lambda *_: self.lr_lbl.configure(text=f"{self.lr_var.get():.2f}"))

        ep_row = tb.Frame(train_tab)
        ep_row.pack(fill=X, pady=(10, 0))
        tb.Label(ep_row, text="Epochs", width=12).pack(side=LEFT)
        self.epochs_var = tk.IntVar(value=1000)
        self.epochs_scale = tb.Scale(ep_row, from_=100, to=5000, variable=self.epochs_var)
        self.epochs_scale.pack(side=LEFT, fill=X, expand=True)
        self.epochs_lbl = tb.Label(ep_row, text="1000")
        self.epochs_lbl.pack(side=LEFT, padx=(8, 0))
        self.epochs_var.trace_add("write", lambda *_: self.epochs_lbl.configure(text=str(int(self.epochs_var.get()))))

        self.progress = tb.Progressbar(train_tab, maximum=100)
        self.progress.pack(fill=X, pady=(14, 6))

        tb.Button(train_tab, text="Train", command=self.train, bootstyle="primary").pack(anchor="w")

        # Dataset
        btns = tb.Frame(data_tab)
        btns.pack(fill=X)
        tb.Button(btns, text="Load", command=self.load_dataset, bootstyle="secondary").pack(side=LEFT, padx=(0, 6))
        tb.Button(btns, text="Save", command=self.save_dataset, bootstyle="secondary").pack(side=LEFT, padx=(0, 6))
        tb.Button(btns, text="Clear Data", command=self.clear_dataset, bootstyle="danger").pack(side=LEFT)

        self.dataset_info = ScrolledText(data_tab, height=12, font=("Consolas", 9))
        self.dataset_info.pack(fill=BOTH, expand=True, pady=(10, 0))
        self.dataset_info.configure(state=DISABLED)

        # Log
        self.log = ScrolledText(log_tab, height=20, font=("Consolas", 9))
        self.log.pack(fill=BOTH, expand=True)
        self.log.configure(state=DISABLED)

        self._init_canvas_cells()

    def _init_canvas_cells(self) -> None:
        self.canvas.delete("all")
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                x1, y1 = c * CELL_SIZE, r * CELL_SIZE
                x2, y2 = x1 + CELL_SIZE, y1 + CELL_SIZE
                rid = self.canvas.create_rectangle(x1, y1, x2, y2, fill="white", outline="#C8C8C8")
                self._rect_ids[r][c] = rid

    def _paint_cell_at(self, x: int, y: int) -> None:
        col = x // CELL_SIZE
        row = y // CELL_SIZE
        if 0 <= row < GRID_ROWS and 0 <= col < GRID_COLS:
            self.grid.set(row, col, 1)
            rid = self._rect_ids[row][col]
            self.canvas.itemconfigure(rid, fill="black")

    def _on_down(self, e: tk.Event) -> None:
        self._dragging = True
        self._paint_cell_at(e.x, e.y)

    def _on_drag(self, e: tk.Event) -> None:
        if self._dragging:
            self._paint_cell_at(e.x, e.y)

    def _on_up(self, _e: tk.Event) -> None:
        self._dragging = False

    def _log(self, msg: str) -> None:
        self.log.configure(state=NORMAL)
        self.log.insert(END, msg + "\n")
        self.log.see(END)
        self.log.configure(state=DISABLED)

    def _refresh_dataset_info(self) -> None:
        self.dataset_info.configure(state=NORMAL)
        self.dataset_info.delete("1.0", END)
        self.dataset_info.insert("1.0", self.dataset.stats_text())
        self.dataset_info.configure(state=DISABLED)

    def _refresh_model_controls(self) -> None:
        is_mlp = self.model_var.get() == "MLP"
        if is_mlp:
            self.hidden_row.pack(fill=X, pady=(10, 0))
        else:
            self.hidden_row.forget()

    def _clear_prediction_ui(self) -> None:
        self.result_var.set("—")
        self.conf_var.set("")
        self.top3_text.configure(state=NORMAL)
        self.top3_text.delete("1.0", END)
        self.top3_text.configure(state=DISABLED)

    # ---------- Actions ----------

    def clear_canvas(self) -> None:
        self.grid.clear()
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                self.canvas.itemconfigure(self._rect_ids[r][c], fill="white")
        self._clear_prediction_ui()

    def add_sample(self) -> None:
        try:
            label = self.label_entry.get().strip()
            self.dataset.add(label, self.grid.to_flat())
            self._refresh_dataset_info()
            self._log(f"✓ Added '{label}' to dataset ({len(self.dataset.samples)} samples)")
            self.label_entry.delete(0, END)
            self.clear_canvas()
        except Exception as e:
            messagebox.showerror("Add Sample", str(e))

    def _active_model(self):
        return self.model_single if self.model_var.get() == "Single" else self.model_mlp

    def train(self) -> None:
        if len(self.dataset.samples) < 2 or len(self.dataset.label_mapping) < 1:
            messagebox.showerror("Train", "Need at least 2 samples to train.")
            return

        model_kind = self.model_var.get()
        lr = float(self.lr_var.get())
        epochs = int(self.epochs_var.get())
        hidden = int(self.hidden_var.get())

        X = np.array(self.dataset.samples, dtype=float)
        output_size = len(self.dataset.label_mapping)
        y = np.zeros((len(self.dataset.labels), output_size), dtype=float)
        for i, label_idx in enumerate(self.dataset.labels):
            y[i, int(label_idx)] = 1.0

        self._log("=" * 50)
        self._log("STARTING TRAINING")
        self._log("=" * 50)

        if model_kind == "Single":
            self.model_single = SinglePerceptron(self.dataset.input_size, output_size, lr)
            self.model_mlp = None
            self._log(f"Model: Single")
            self._log(f"Network: {self.dataset.input_size} → {output_size}")
        else:
            self.model_mlp = MultiLayerPerceptron(self.dataset.input_size, hidden, output_size, lr)
            self.model_single = None
            self._log(f"Model: MLP")
            self._log(f"Network: {self.dataset.input_size} → {hidden} → {output_size}")

        self._log(f"Learning rate: {lr}")
        self._log(f"Epochs: {epochs}")
        self._log(f"Training samples: {len(X)}")

        self.progress.configure(value=0)
        self.root.update_idletasks()

        log_interval = max(1, epochs // 10)
        errors: List[float] = []

        for ep in range(epochs):
            if self.model_single is not None:
                err = self.model_single.train_step(X, y)
            else:
                err = self.model_mlp.train_step(X, y)  # type: ignore[union-attr]
            errors.append(err)

            self.progress.configure(value=(ep + 1) / epochs * 100)
            if (ep + 1) % log_interval == 0 or ep == 0:
                self._log(f"Epoch {ep + 1}/{epochs} - Error: {err:.6f}")
            self.root.update()

        if self.model_single is not None:
            preds = self.model_single.predict(X)
        else:
            preds = self.model_mlp.predict(X)  # type: ignore[union-attr]

        acc = float(np.mean(np.argmax(preds, axis=1) == np.array(self.dataset.labels))) * 100.0

        self._log("=" * 50)
        self._log("TRAINING COMPLETE")
        self._log("=" * 50)
        self._log(f"Final Error: {errors[-1]:.6f}")
        self._log(f"Training Accuracy: {acc:.2f}%")

    def recognize(self) -> None:
        model = self._active_model()
        if model is None:
            messagebox.showerror("Recognize", "Train the selected model first.")
            return

        vec = self.grid.to_flat().reshape(1, -1)
        if int(np.sum(vec)) == 0:
            messagebox.showwarning("Recognize", "Draw a digit first.")
            return

        pred = model.predict(vec)[0]
        idx = int(np.argmax(pred))
        conf = float(pred[idx]) * 100.0
        letter = self.dataset.reverse_label_mapping.get(idx, "?")

        self.result_var.set(letter)
        self.conf_var.set(f"Confidence: {conf:.1f}%")

        top3 = np.argsort(pred)[-3:][::-1]
        lines = []
        for rank, i in enumerate(top3, start=1):
            ltr = self.dataset.reverse_label_mapping.get(int(i), "?")
            lines.append(f"{rank}. {ltr}: {float(pred[int(i)]) * 100.0:.1f}%")

        self.top3_text.configure(state=NORMAL)
        self.top3_text.delete("1.0", END)
        self.top3_text.insert("1.0", "\n".join(lines))
        self.top3_text.configure(state=DISABLED)

        self._log(f"✓ Recognized as '{letter}' ({conf:.1f}% confidence)")

    def save_dataset(self) -> None:
        if not self.dataset.samples:
            messagebox.showinfo("Save", "No data to save.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            initialfile=DEFAULT_DATASET_FILE,
        )
        if not path:
            return

        try:
            self.dataset.save(path)
            self._log(f"✓ Dataset saved to: {path}")
        except Exception as e:
            messagebox.showerror("Save", str(e))

    def load_dataset(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Text files", "*.txt"), ("All files", "*.*")])
        if not path:
            return

        try:
            self.dataset = Dataset.load(path)
            self.model_single = None
            self.model_mlp = None
            self._refresh_dataset_info()
            self._clear_prediction_ui()
            self._log(f"✓ Dataset loaded from: {path} ({len(self.dataset.samples)} samples)")
        except Exception as e:
            messagebox.showerror("Load", str(e))

    def clear_dataset(self) -> None:
        if not messagebox.askyesno("Confirm", "Clear all training data?"):
            return
        self.dataset = Dataset()
        self.model_single = None
        self.model_mlp = None
        self._refresh_dataset_info()
        self._clear_prediction_ui()
        self._log("✓ All data cleared")

    def _load_default_dataset_if_present(self) -> None:
        if os.path.exists(DEFAULT_DATASET_FILE):
            try:
                self.dataset = Dataset.load(DEFAULT_DATASET_FILE)
                self._log(f"✓ Loaded default dataset: {DEFAULT_DATASET_FILE} ({len(self.dataset.samples)} samples)")
            except Exception as e:
                self._log(f"⚠ Could not load default dataset: {e}")

    def run(self) -> None:
        self.root.mainloop()


if __name__ == "__main__":
    App().run()
