"""Fuzzy Image Processing: Adaptive Sharpening (standalone)

Uses OpenCV + a small fuzzy logic system to choose an unsharp-mask amount based on:
- blur score (variance of Laplacian)
- noise score (high-frequency energy)

Run (webcam only):
    python fuzzy_img_sharpen.py
    python fuzzy_img_sharpen.py --camera 1

Controls:
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
# Sharpening logic
# ----------------------------


def build_system() -> MamdaniSystem:
    # blur_var: variance of Laplacian; typical range 0..2000+ depending on image.
    blur = FuzzyVariable(
        name="blur",
        start=0.0,
        end=2500.0,
        step=10.0,
        sets={
            "very_blurry": trapmf(0.0, 0.0, 80.0, 180.0),
            "blurry": trimf(120.0, 260.0, 550.0),
            "sharp": trapmf(450.0, 800.0, 2500.0, 2500.0),
        },
    )

    # noise_score: std of high-frequency residual (0..40-ish)
    noise = FuzzyVariable(
        name="noise",
        start=0.0,
        end=50.0,
        step=0.5,
        sets={
            "low": trapmf(0.0, 0.0, 6.0, 12.0),
            "medium": trimf(10.0, 18.0, 28.0),
            "high": trapmf(24.0, 32.0, 50.0, 50.0),
        },
    )

    # amount: unsharp amount 0..1.5
    amount = FuzzyVariable(
        name="amount",
        start=0.0,
        end=1.5,
        step=0.01,
        sets={
            "none": trapmf(0.0, 0.0, 0.05, 0.15),
            "small": trimf(0.10, 0.30, 0.55),
            "medium": trimf(0.40, 0.70, 1.00),
            "strong": trapmf(0.90, 1.10, 1.50, 1.50),
        },
    )

    rules = [
        # More blur => stronger sharpening.
        FuzzyRule([("blur", "very_blurry"), ("noise", "low")], ("amount", "strong")),
        FuzzyRule([("blur", "very_blurry"), ("noise", "medium")], ("amount", "medium")),
        FuzzyRule([("blur", "very_blurry"), ("noise", "high")], ("amount", "small")),

        FuzzyRule([("blur", "blurry"), ("noise", "low")], ("amount", "medium")),
        FuzzyRule([("blur", "blurry"), ("noise", "medium")], ("amount", "small")),
        FuzzyRule([("blur", "blurry"), ("noise", "high")], ("amount", "small")),

        # Already sharp => minimal sharpening (especially if noisy).
        FuzzyRule([("blur", "sharp"), ("noise", "high")], ("amount", "none")),
        FuzzyRule([("blur", "sharp"), ("noise", "medium")], ("amount", "none")),
        FuzzyRule([("blur", "sharp"), ("noise", "low")], ("amount", "small")),
    ]

    return MamdaniSystem(inputs={"blur": blur, "noise": noise}, outputs={"amount": amount}, rules=rules)


def compute_features(bgr: np.ndarray) -> Tuple[float, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    lap = cv2.Laplacian(gray, cv2.CV_64F)
    blur_var = float(lap.var())

    blur_img = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
    residual = gray.astype(np.float32) - blur_img.astype(np.float32)
    noise_score = float(np.std(residual))

    return blur_var, noise_score


def apply_unsharp(bgr: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0.01:
        return bgr.copy()
    blurred = cv2.GaussianBlur(bgr, (0, 0), sigmaX=1.0)
    sharp = cv2.addWeighted(bgr, 1.0 + float(amount), blurred, -float(amount), 0)
    return sharp


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fuzzy Adaptive Sharpening (OpenCV)")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    p.add_argument("--mirror", action="store_true", help="Mirror the webcam image")
    p.add_argument("--output", help="Snapshot path when pressing 's'")
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

            blur_var, noise_score = compute_features(frame)
            out = system.infer({"blur": blur_var, "noise": noise_score})
            amount = float(out["amount"])
            sharpened = apply_unsharp(frame, amount)

            sharpened = _overlay_text(
                sharpened,
                [
                    f"lapvar={blur_var:.1f} noise={noise_score:.2f}",
                    f"amount={amount:.3f}",
                    "q/ESC: quit   s: save",
                ],
            )
            last_processed = sharpened

            if not no_show:
                cv2.imshow("Webcam - Original", frame)
                cv2.imshow("Webcam - Sharpened", sharpened)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s"):
                    path = output_path or "fuzzy_sharpen_snapshot.png"
                    cv2.imwrite(path, last_processed)
                    print(f"saved_snapshot={path}")
            else:
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
    run_webcam(args.camera, args.mirror, args.output, args.no_show)


if __name__ == "__main__":
    main()
