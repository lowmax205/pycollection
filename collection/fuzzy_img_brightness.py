"""Fuzzy Image Processing: Brightness Correction (standalone)

Uses OpenCV + a small fuzzy logic system to choose a gamma value based on:
- mean brightness
- dark pixel ratio

Run (webcam, default):
    python fuzzy_img_brightness.py
    python fuzzy_img_brightness.py --camera 1

Run (image file):
    python fuzzy_img_brightness.py --input path/to/image.jpg
    python fuzzy_img_brightness.py --input in.jpg --output out.jpg

Controls (webcam):
- Press 'q' or ESC to quit
- Press 's' to save a snapshot (uses --output if provided)

Requires: opencv-python
  pip install opencv-python
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple


try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Missing dependency. Install with: pip install opencv-python numpy\n"
        f"Import error: {e}"
    )


# ----------------------------
# Minimal fuzzy engine (Mamdani + centroid)
# ----------------------------

MembershipFn = Callable[[float], float]


def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def trimf(a: float, b: float, c: float) -> MembershipFn:
    def f(x: float) -> float:
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return clamp01((x - a) / (b - a))
        return clamp01((c - x) / (c - b))

    return f


def trapmf(a: float, b: float, c: float, d: float) -> MembershipFn:
    def f(x: float) -> float:
        if x <= a or x >= d:
            return 0.0
        if b <= x <= c:
            return 1.0
        if a < x < b:
            return clamp01((x - a) / (b - a))
        return clamp01((d - x) / (d - c))

    return f


@dataclass(frozen=True)
class FuzzyVariable:
    name: str
    start: float
    end: float
    step: float
    sets: Mapping[str, MembershipFn]

    def universe(self) -> List[float]:
        n = int(round((self.end - self.start) / self.step))
        return [self.start + i * self.step for i in range(n + 1)]


@dataclass(frozen=True)
class FuzzyRule:
    antecedents: Sequence[Tuple[str, str]]
    consequent: Tuple[str, str]


def centroid(xs: Sequence[float], mus: Sequence[float]) -> float:
    num = 0.0
    den = 0.0
    for x, mu in zip(xs, mus):
        num += x * mu
        den += mu
    if den == 0.0:
        return float(xs[len(xs) // 2])
    return num / den


class MamdaniSystem:
    def __init__(
        self,
        inputs: Mapping[str, FuzzyVariable],
        outputs: Mapping[str, FuzzyVariable],
        rules: Sequence[FuzzyRule],
    ) -> None:
        self.inputs = dict(inputs)
        self.outputs = dict(outputs)
        self.rules = list(rules)

    def fuzzify(self, crisp_inputs: Mapping[str, float]) -> Dict[str, Dict[str, float]]:
        degrees: Dict[str, Dict[str, float]] = {}
        for var_name, var in self.inputs.items():
            x = float(crisp_inputs[var_name])
            degrees[var_name] = {set_name: mf(x) for set_name, mf in var.sets.items()}
        return degrees

    def infer(self, crisp_inputs: Mapping[str, float]) -> Dict[str, float]:
        in_deg = self.fuzzify(crisp_inputs)
        results: Dict[str, float] = {}

        for out_name, out_var in self.outputs.items():
            xs = out_var.universe()
            aggregated = [0.0 for _ in xs]

            for rule in self.rules:
                cons_var, cons_set = rule.consequent
                if cons_var != out_name:
                    continue

                firing = 1.0
                for ant_var, ant_set in rule.antecedents:
                    firing = min(firing, in_deg[ant_var][ant_set])
                    if firing <= 0.0:
                        break

                if firing <= 0.0:
                    continue

                out_mf = out_var.sets[cons_set]
                for i, x in enumerate(xs):
                    aggregated[i] = max(aggregated[i], min(firing, out_mf(x)))

            results[out_name] = centroid(xs, aggregated)

        return results


# ----------------------------
# Brightness correction logic
# ----------------------------


def build_system() -> MamdaniSystem:
    # brightness: mean gray 0..255
    brightness = FuzzyVariable(
        name="brightness",
        start=0.0,
        end=255.0,
        step=1.0,
        sets={
            "dark": trapmf(0.0, 0.0, 50.0, 110.0),
            "normal": trimf(90.0, 128.0, 170.0),
            "bright": trapmf(150.0, 200.0, 255.0, 255.0),
        },
    )

    # dark_ratio: fraction of pixels below threshold (0..1)
    dark_ratio = FuzzyVariable(
        name="dark_ratio",
        start=0.0,
        end=1.0,
        step=0.01,
        sets={
            "low": trapmf(0.0, 0.0, 0.10, 0.25),
            "medium": trimf(0.15, 0.35, 0.55),
            "high": trapmf(0.45, 0.65, 1.0, 1.0),
        },
    )

    # gamma: we use output range 0.6..1.8
    # gamma < 1 brightens, gamma > 1 darkens (with our LUT formula)
    gamma = FuzzyVariable(
        name="gamma",
        start=0.60,
        end=1.80,
        step=0.01,
        sets={
            "brighten_lot": trapmf(0.60, 0.60, 0.75, 0.95),
            "brighten": trimf(0.85, 1.00, 1.15),
            "neutral": trimf(0.95, 1.10, 1.25),
            "darken": trapmf(1.15, 1.35, 1.80, 1.80),
        },
    )

    rules = [
        # If image is dark or has lots of dark pixels => brighten.
        FuzzyRule([("brightness", "dark")], ("gamma", "brighten_lot")),
        FuzzyRule([("dark_ratio", "high")], ("gamma", "brighten_lot")),
        FuzzyRule([("brightness", "normal"), ("dark_ratio", "medium")], ("gamma", "brighten")),

        # If image is already bright => darken slightly.
        FuzzyRule([("brightness", "bright")], ("gamma", "darken")),

        # Otherwise keep near-neutral.
        FuzzyRule([("brightness", "normal"), ("dark_ratio", "low")], ("gamma", "neutral")),
    ]

    return MamdaniSystem(inputs={"brightness": brightness, "dark_ratio": dark_ratio}, outputs={"gamma": gamma}, rules=rules)


def compute_features(bgr: np.ndarray) -> Tuple[float, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))
    dark_ratio = float(np.mean(gray < 60))
    return mean_brightness, dark_ratio


def apply_gamma(bgr: np.ndarray, gamma: float) -> np.ndarray:
    # LUT formula: out = (in/255)^(1/gamma) * 255
    # gamma < 1 -> brighter, gamma > 1 -> darker
    inv_gamma = 1.0 / float(gamma)
    table = np.array([((i / 255.0) ** inv_gamma) * 255.0 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(bgr, table)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fuzzy Brightness Correction (OpenCV)")
    p.add_argument("--input", help="Input image path (optional; if omitted uses webcam)")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    p.add_argument("--mirror", action="store_true", help="Mirror the webcam image")
    p.add_argument("--output", help="Output image path (image mode), or snapshot path (webcam mode)")
    p.add_argument("--no-show", action="store_true", help="Do not open preview windows")
    return p


def _overlay_text(img: np.ndarray, lines: Sequence[str]) -> np.ndarray:
    out = img.copy()
    y = 24
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
        y += 24
    return out


def run_webcam(camera_index: int, mirror: bool, output_path: str | None, no_show: bool) -> None:
    if no_show and not output_path:
        raise SystemExit("Webcam mode with --no-show needs --output (snapshot path).")

    cap = cv2.VideoCapture(int(camera_index))
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {camera_index}")

    system = build_system()
    last_processed: np.ndarray | None = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            if mirror:
                frame = cv2.flip(frame, 1)

            mean_brightness, dark_ratio = compute_features(frame)
            out = system.infer({"brightness": mean_brightness, "dark_ratio": dark_ratio})
            gamma = float(out["gamma"])
            corrected = apply_gamma(frame, gamma)

            corrected = _overlay_text(
                corrected,
                [
                    f"mean={mean_brightness:.1f} dark_ratio={dark_ratio:.3f}",
                    f"gamma={gamma:.3f}",
                    "q/ESC: quit   s: save",
                ],
            )
            last_processed = corrected

            if not no_show:
                cv2.imshow("Webcam - Original", frame)
                cv2.imshow("Webcam - Brightness Corrected", corrected)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    path = output_path or "fuzzy_brightness_snapshot.png"
                    cv2.imwrite(path, last_processed)
                    print(f"saved_snapshot={path}")
            else:
                # no-show mode: take a single snapshot and exit
                if last_processed is not None:
                    cv2.imwrite(output_path, last_processed)  # type: ignore[arg-type]
                    print(f"saved_snapshot={output_path}")
                break
    finally:
        cap.release()
        if not no_show:
            cv2.destroyAllWindows()


def main() -> None:
    args = build_parser().parse_args()

    if not args.input:
        run_webcam(args.camera, args.mirror, args.output, args.no_show)
        return

    img = cv2.imread(args.input)
    if img is None:
        raise SystemExit(f"Could not read image: {args.input}")

    system = build_system()
    mean_brightness, dark_ratio = compute_features(img)

    out = system.infer({"brightness": mean_brightness, "dark_ratio": dark_ratio})
    gamma = float(out["gamma"])

    corrected = apply_gamma(img, gamma)

    print("--- Fuzzy Brightness Correction ---")
    print(f"mean_brightness={mean_brightness:.2f}  dark_ratio={dark_ratio:.3f}")
    print(f"chosen_gamma={gamma:.3f}")

    if args.output:
        ok = cv2.imwrite(args.output, corrected)
        if not ok:
            raise SystemExit(f"Could not write output: {args.output}")
        print(f"saved={args.output}")

    if not args.no_show and not args.output:
        cv2.imshow("Original", img)
        cv2.imshow("Corrected", corrected)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
