"""
Combined (Single + Multi) digit recognition app using PySimpleGUI.

This script replaces separate single/multi scripts by offering a model toggle:
- "Single" = single-layer perceptron (sigmoid)
- "MLP"    = one-hidden-layer perceptron (sigmoid hidden + sigmoid output)

Dataset format is compatible with the original scripts (JSON stored in *.txt):
    {
        "samples": [[...], ...],
        "labels": [0, 1, ...],
        "label_mapping": {"0": 0, "1": 1, ...},
        "reverse_label_mapping": {"0": "0", "1": "1", ...},
        "grid_rows": 7,
        "grid_cols": 5
    }

Dependencies:
  pip install numpy PySimpleGUI

Run:
  python perceptron.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Note: the public PyPI `PySimpleGUI` package may be a limited/stub package.
# We prefer the open-source drop-in fork when available.
try:
    import FreeSimpleGUI as sg  # type: ignore
except Exception:
    try:
        import PySimpleGUI as sg  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "A PySimpleGUI-compatible library is required.\n"
            "Install one of:\n"
            "  python -m pip install --upgrade FreeSimpleGUI\n"
            "  (or) install official PySimpleGUI from their private index."
        ) from exc

if not hasattr(sg, "Window") or not hasattr(sg, "Graph"):
    raise SystemExit(
        "The imported PySimpleGUI package is missing required APIs (Window/Graph).\n"
        "Fix by installing FreeSimpleGUI:\n"
        "  python -m pip install --upgrade FreeSimpleGUI\n\n"
        "Optional: remove the stub PySimpleGUI to silence messages:\n"
        "  python -m pip uninstall PySimpleGUI"
    )


GRID_ROWS = 7
GRID_COLS = 5
PIXEL_SIZE = 28
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
        if self.samples is None:
            self.samples = []
        if self.labels is None:
            self.labels = []
        if self.label_mapping is None:
            self.label_mapping = {}
        if self.reverse_label_mapping is None:
            self.reverse_label_mapping = {}

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

    def set_cell(self, row: int, col: int, value: int) -> None:
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.pixels[row][col] = 1 if value else 0

    def to_flat(self) -> np.ndarray:
        return np.array([p for row in self.pixels for p in row], dtype=float)


def _graph_size() -> Tuple[int, int]:
    return GRID_COLS * PIXEL_SIZE, GRID_ROWS * PIXEL_SIZE


def _draw_grid(graph: "sg.Graph", grid: PixelGrid) -> None:
    w, h = _graph_size()
    graph.erase()

    # filled cells
    for r in range(grid.rows):
        for c in range(grid.cols):
            if grid.pixels[r][c] == 1:
                x1 = c * PIXEL_SIZE
                y1 = (grid.rows - 1 - r) * PIXEL_SIZE
                x2 = x1 + PIXEL_SIZE
                y2 = y1 + PIXEL_SIZE
                graph.draw_rectangle((x1, y1), (x2, y2), fill_color="black", line_color="black")

    # grid lines
    for i in range(grid.rows + 1):
        y = i * PIXEL_SIZE
        graph.draw_line((0, y), (w, y), color="#C8C8C8")
    for j in range(grid.cols + 1):
        x = j * PIXEL_SIZE
        graph.draw_line((x, 0), (x, h), color="#C8C8C8")


def _pos_to_cell(x: float, y: float) -> Tuple[int, int]:
    col = int(x // PIXEL_SIZE)
    row_from_bottom = int(y // PIXEL_SIZE)
    row = GRID_ROWS - 1 - row_from_bottom
    return row, col


def run_app() -> None:
    if hasattr(sg, "theme"):
        sg.theme("LightBlue3")

    w, h = _graph_size()
    graph = sg.Graph(
        canvas_size=(w, h),
        graph_bottom_left=(0, 0),
        graph_top_right=(w, h),
        background_color="white",
        enable_events=True,
        drag_submits=True,
        key="-GRAPH-",
    )

    # Left: drawing
    left_col = [
        [sg.Text("Digit Recognition", font=("Segoe UI", 14, "bold"))],
        [sg.Text("Draw a digit (drag with mouse):")],
        [graph],
        [
            sg.Button("Clear", key="-CLEAR-", size=(10, 1)),
            sg.Button("Recognize", key="-RECOG-", size=(12, 1)),
        ],
        [sg.Frame("Add Sample", [[sg.Input(key="-LABEL-", size=(8, 1)), sg.Button("Add", key="-ADD-")]])],
        [
            sg.Frame(
                "Result",
                [
                    [sg.Text("Digit:", size=(7, 1)), sg.Text("—", key="-RESULT-", font=("Segoe UI", 24, "bold"))],
                    [sg.Text("Confidence:", size=(10, 1)), sg.Text("", key="-CONF-")],
                    [sg.Text("Top-3:")],
                    [sg.Multiline("", key="-TOP3-", size=(22, 4), disabled=True, no_scrollbar=True)],
                ],
            )
        ],
    ]

    # Right: training + dataset + log
    model_choices = ["Single", "MLP"]

    train_frame = sg.Frame(
        "Training",
        [
            [sg.Text("Model"), sg.Combo(model_choices, default_value="Single", readonly=True, enable_events=True, key="-MODEL-")],
            [sg.Text("Hidden size", key="-HIDDENLBL-"), sg.Slider((5, 30), default_value=10, resolution=1, orientation="h", key="-HIDDEN-", size=(25, 15), visible=False)],
            [sg.Text("Learning rate"), sg.Slider((0.01, 1.0), default_value=0.10, resolution=0.01, orientation="h", key="-LR-", size=(25, 15))],
            [sg.Text("Epochs"), sg.Slider((100, 5000), default_value=1000, resolution=50, orientation="h", key="-EPOCHS-", size=(25, 15))],
            [sg.ProgressBar(1000, orientation="h", size=(30, 15), key="-PROG-")],
            [sg.Button("Train", key="-TRAIN-", size=(10, 1))],
        ],
    )

    dataset_frame = sg.Frame(
        "Dataset",
        [
            [sg.Button("Load", key="-LOAD-"), sg.Button("Save", key="-SAVE-"), sg.Button("Clear Data", key="-CLEARDS-")],
            [sg.Multiline("No data loaded", key="-DSINFO-", size=(42, 9), disabled=True)],
        ],
    )

    log_frame = sg.Frame("Log", [[sg.Multiline("", key="-LOG-", size=(42, 12), autoscroll=True, disabled=True)]])

    right_col = [[train_frame], [dataset_frame], [log_frame]]

    layout = [[sg.Column(left_col, vertical_alignment="top"), sg.VSeparator(), sg.Column(right_col, vertical_alignment="top")]]

    window = sg.Window("Digit Recognition", layout, finalize=True)

    grid_state = PixelGrid(GRID_ROWS, GRID_COLS)
    dataset = Dataset()
    model_single: Optional[SinglePerceptron] = None
    model_mlp: Optional[MultiLayerPerceptron] = None

    def log(msg: str) -> None:
        window["-LOG-"].update(disabled=False)
        window["-LOG-"].print(msg)
        window["-LOG-"].update(disabled=True)

    def refresh_ds() -> None:
        window["-DSINFO-"].update(disabled=False)
        window["-DSINFO-"].update(dataset.stats_text())
        window["-DSINFO-"].update(disabled=True)

    def refresh_model_ui() -> None:
        is_mlp = (window["-MODEL-"].get() == "MLP")
        window["-HIDDEN-"].update(visible=is_mlp)
        window["-HIDDENLBL-"].update(visible=is_mlp)
        window.refresh()

    def clear_prediction_ui() -> None:
        window["-RESULT-"].update("—")
        window["-CONF-"].update("")
        window["-TOP3-"].update("")

    def default_load_if_present() -> None:
        nonlocal dataset
        if os.path.exists(DEFAULT_DATASET_FILE):
            try:
                dataset = Dataset.load(DEFAULT_DATASET_FILE)
                refresh_ds()
                log(f"✓ Loaded default dataset: {DEFAULT_DATASET_FILE} ({len(dataset.samples)} samples)")
            except Exception as e:
                log(f"⚠ Could not load default dataset: {e}")

    _draw_grid(window["-GRAPH-"], grid_state)
    refresh_ds()
    refresh_model_ui()
    default_load_if_present()

    while True:
        event, values = window.read(timeout=50)
        if event in (sg.WIN_CLOSED, "Exit"):
            break

        if event == "-MODEL-":
            refresh_model_ui()

        if event == "-GRAPH-":
            pos = values.get("-GRAPH-")
            if pos is not None:
                x, y = pos
                row, col = _pos_to_cell(x, y)
                grid_state.set_cell(row, col, 1)
                _draw_grid(window["-GRAPH-"], grid_state)

        if event == "-CLEAR-":
            grid_state.clear()
            _draw_grid(window["-GRAPH-"], grid_state)
            clear_prediction_ui()

        if event == "-ADD-":
            try:
                label = (values.get("-LABEL-") or "").strip().upper()
                dataset.add(label, grid_state.to_flat())
                refresh_ds()
                log(f"✓ Added '{label}' to dataset ({len(dataset.samples)} samples)")
                window["-LABEL-"].update("")
                grid_state.clear()
                _draw_grid(window["-GRAPH-"], grid_state)
                clear_prediction_ui()
            except Exception as e:
                sg.popup_error(str(e), title="Add Sample")

        if event == "-TRAIN-":
            if len(dataset.samples) < 2 or len(dataset.label_mapping) < 1:
                sg.popup_error("Need at least 2 samples to train.", title="Train")
                continue

            model_kind = values["-MODEL-"]
            lr = float(values["-LR-"])
            epochs = int(values["-EPOCHS-"])
            hidden = int(values["-HIDDEN-"]) if model_kind == "MLP" else 0

            X = np.array(dataset.samples, dtype=float)
            output_size = len(dataset.label_mapping)
            y = np.zeros((len(dataset.labels), output_size), dtype=float)
            for i, label_idx in enumerate(dataset.labels):
                y[i, int(label_idx)] = 1.0

            log("=" * 50)
            log("STARTING TRAINING")
            log("=" * 50)

            if model_kind == "Single":
                model_single = SinglePerceptron(dataset.input_size, output_size, lr)
                model_mlp = None
                log(f"Model: Single")
                log(f"Network: {dataset.input_size} → {output_size}")
            else:
                model_mlp = MultiLayerPerceptron(dataset.input_size, hidden, output_size, lr)
                model_single = None
                log(f"Model: MLP")
                log(f"Network: {dataset.input_size} → {hidden} → {output_size}")

            log(f"Learning rate: {lr}")
            log(f"Epochs: {epochs}")
            log(f"Training samples: {len(X)}")

            window["-PROG-"].update(current_count=0, max=epochs)
            log_interval = max(1, epochs // 10)
            errors: List[float] = []

            for ep in range(epochs):
                if model_single is not None:
                    err = model_single.train_step(X, y)
                else:
                    err = model_mlp.train_step(X, y)  # type: ignore[union-attr]
                errors.append(err)
                window["-PROG-"].update(current_count=ep + 1)
                if (ep + 1) % log_interval == 0 or ep == 0:
                    log(f"Epoch {ep + 1}/{epochs} - Error: {err:.6f}")
                window.refresh()

            if model_single is not None:
                preds = model_single.predict(X)
            else:
                preds = model_mlp.predict(X)  # type: ignore[union-attr]

            acc = float(np.mean(np.argmax(preds, axis=1) == np.array(dataset.labels))) * 100.0

            log("=" * 50)
            log("TRAINING COMPLETE")
            log("=" * 50)
            log(f"Final Error: {errors[-1]:.6f}")
            log(f"Training Accuracy: {acc:.2f}%")

        if event == "-RECOG-":
            model_kind = values["-MODEL-"]
            active_model = model_single if model_kind == "Single" else model_mlp
            if active_model is None:
                sg.popup_error("Train the selected model first.", title="Recognize")
                continue

            vec = grid_state.to_flat().reshape(1, -1)
            if int(np.sum(vec)) == 0:
                sg.popup("Draw a digit first.", title="Recognize")
                continue

            pred = active_model.predict(vec)[0]
            idx = int(np.argmax(pred))
            conf = float(pred[idx]) * 100.0
            letter = dataset.reverse_label_mapping.get(idx, "?")

            window["-RESULT-"].update(letter)
            window["-CONF-"].update(f"{conf:.1f}%")

            top3 = np.argsort(pred)[-3:][::-1]
            lines = []
            for rank, i in enumerate(top3, start=1):
                ltr = dataset.reverse_label_mapping.get(int(i), "?")
                lines.append(f"{rank}. {ltr}: {float(pred[int(i)]) * 100.0:.1f}%")
            window["-TOP3-"].update("\n".join(lines))
            log(f"✓ Recognized as '{letter}' ({conf:.1f}% confidence)")

        if event == "-SAVE-":
            if not dataset.samples:
                sg.popup("No data to save.", title="Save")
                continue
            path = sg.popup_get_file(
                "Save Dataset",
                save_as=True,
                default_extension=".txt",
                file_types=(("Text", "*.txt"), ("All", "*.*")),
                initial_folder=os.getcwd(),
                default_path=DEFAULT_DATASET_FILE,
            )
            if path:
                try:
                    dataset.save(path)
                    log(f"✓ Dataset saved to: {path}")
                except Exception as e:
                    sg.popup_error(str(e), title="Save")

        if event == "-LOAD-":
            path = sg.popup_get_file(
                "Load Dataset",
                file_types=(("Text", "*.txt"), ("All", "*.*")),
                initial_folder=os.getcwd(),
            )
            if path:
                try:
                    dataset = Dataset.load(path)
                    refresh_ds()
                    model_single = None
                    model_mlp = None
                    clear_prediction_ui()
                    log(f"✓ Dataset loaded from: {path} ({len(dataset.samples)} samples)")
                except Exception as e:
                    sg.popup_error(str(e), title="Load")

        if event == "-CLEARDS-":
            if sg.popup_yes_no("Clear all training data?", title="Confirm") == "Yes":
                dataset = Dataset()
                model_single = None
                model_mlp = None
                refresh_ds()
                clear_prediction_ui()
                log("✓ All data cleared")

    window.close()


if __name__ == "__main__":
    run_app()
