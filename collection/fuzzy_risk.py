"""Fuzzy Risk Assessment (standalone).

Run:
  python fuzzy_risk.py
  python fuzzy_risk.py --likelihood 7 --impact 9

Inputs:
- likelihood: 0..10
- impact: 0..10
Output:
- risk score: 0..10 and a label (LOW/MEDIUM/HIGH/CRITICAL)

No third-party dependencies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Dict, List, Mapping, Sequence, Tuple

import tkinter as tk
from tkinter import ttk

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
    """Triangular membership function."""

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
    """Trapezoidal membership function."""

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
    antecedents: Sequence[Tuple[str, str]]  # (var_name, set_name)
    consequent: Tuple[str, str]  # (out_var_name, out_set_name)


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


def build_risk_system() -> MamdaniSystem:
    likelihood = FuzzyVariable(
        name="likelihood",
        start=0.0,
        end=10.0,
        step=0.1,
        sets={
            "low": trapmf(0.0, 0.0, 2.0, 4.0),
            "medium": trimf(3.0, 5.0, 7.0),
            "high": trapmf(6.0, 8.0, 10.0, 10.0),
        },
    )

    impact = FuzzyVariable(
        name="impact",
        start=0.0,
        end=10.0,
        step=0.1,
        sets={
            "low": trapmf(0.0, 0.0, 2.0, 4.0),
            "medium": trimf(3.0, 5.0, 7.0),
            "high": trapmf(6.0, 8.0, 10.0, 10.0),
        },
    )

    risk = FuzzyVariable(
        name="risk",
        start=0.0,
        end=10.0,
        step=0.1,
        sets={
            "low": trapmf(0.0, 0.0, 2.0, 4.0),
            "medium": trimf(3.0, 5.0, 7.0),
            "high": trimf(6.0, 7.5, 9.0),
            "critical": trapmf(8.0, 9.0, 10.0, 10.0),
        },
    )

    rules: List[FuzzyRule] = [
        FuzzyRule([("likelihood", "low"), ("impact", "low")], ("risk", "low")),
        FuzzyRule([("likelihood", "low"), ("impact", "medium")], ("risk", "medium")),
        FuzzyRule([("likelihood", "medium"), ("impact", "low")], ("risk", "medium")),

        FuzzyRule([("likelihood", "medium"), ("impact", "medium")], ("risk", "high")),
        FuzzyRule([("likelihood", "high"), ("impact", "medium")], ("risk", "high")),
        FuzzyRule([("likelihood", "medium"), ("impact", "high")], ("risk", "high")),

        FuzzyRule([("likelihood", "high"), ("impact", "high")], ("risk", "critical")),
        FuzzyRule([("likelihood", "high"), ("impact", "low")], ("risk", "medium")),
        FuzzyRule([("likelihood", "low"), ("impact", "high")], ("risk", "high")),
    ]

    return MamdaniSystem(inputs={"likelihood": likelihood, "impact": impact}, outputs={"risk": risk}, rules=rules)


def risk_label(score: float) -> str:
    if score >= 8.5:
        return "CRITICAL"
    if score >= 6.5:
        return "HIGH"
    if score >= 3.5:
        return "MEDIUM"
    return "LOW"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fuzzy Risk Assessment")
    p.add_argument("--likelihood", type=float, help="Likelihood 0..10")
    p.add_argument("--impact", type=float, help="Impact 0..10")
    p.add_argument("--prompt", action="store_true", help="Prompt for inputs in terminal (no GUI)")
    return p


def run_gui() -> None:
    system = build_risk_system()

    root = tk.Tk()
    root.title("Fuzzy Risk Assessment")
    root.resizable(False, False)

    container = ttk.Frame(root, padding=12)
    container.grid(row=0, column=0, sticky="nsew")

    ttk.Label(container, text="Fuzzy Risk Assessment", font=("Segoe UI", 14, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w"
    )

    likelihood_var = tk.DoubleVar(value=5.0)
    impact_var = tk.DoubleVar(value=5.0)

    ttk.Label(container, text="Likelihood (0–10)").grid(row=1, column=0, sticky="w", pady=(10, 0))
    ttk.Scale(container, from_=0.0, to=10.0, variable=likelihood_var).grid(row=1, column=1, sticky="ew", padx=(10, 10), pady=(10, 0))
    like_lbl = ttk.Label(container, width=6)
    like_lbl.grid(row=1, column=2, sticky="e", pady=(10, 0))

    ttk.Label(container, text="Impact (0–10)").grid(row=2, column=0, sticky="w", pady=(10, 0))
    ttk.Scale(container, from_=0.0, to=10.0, variable=impact_var).grid(row=2, column=1, sticky="ew", padx=(10, 10), pady=(10, 0))
    imp_lbl = ttk.Label(container, width=6)
    imp_lbl.grid(row=2, column=2, sticky="e", pady=(10, 0))

    score_var = tk.StringVar(value="—")
    label_var = tk.StringVar(value="")

    out_box = ttk.Labelframe(container, text="Result", padding=10)
    out_box.grid(row=3, column=0, columnspan=3, sticky="ew", pady=(12, 0))
    ttk.Label(out_box, text="Risk score:").grid(row=0, column=0, sticky="w")
    ttk.Label(out_box, textvariable=score_var, font=("Segoe UI", 16, "bold")).grid(row=0, column=1, sticky="e")
    ttk.Label(out_box, textvariable=label_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def sync_labels() -> None:
        like_lbl.configure(text=f"{likelihood_var.get():.1f}")
        imp_lbl.configure(text=f"{impact_var.get():.1f}")

    def compute() -> None:
        likelihood = float(likelihood_var.get())
        impact = float(impact_var.get())
        out = system.infer({"likelihood": likelihood, "impact": impact})
        r = float(out["risk"])
        score_var.set(f"{r:.2f}")
        label_var.set(f"Level: {risk_label(r)}")

    btn_row = ttk.Frame(container)
    btn_row.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    ttk.Button(btn_row, text="Compute", command=compute).pack(side=tk.LEFT)
    ttk.Button(btn_row, text="Close", command=root.destroy).pack(side=tk.RIGHT)

    container.columnconfigure(1, weight=1)

    likelihood_var.trace_add("write", lambda *_: sync_labels())
    impact_var.trace_add("write", lambda *_: sync_labels())
    sync_labels()
    compute()

    root.mainloop()


def main() -> None:
    args = build_parser().parse_args()

    # If user passes any values or requests prompting, run CLI mode.
    if args.prompt or args.likelihood is not None or args.impact is not None:
        system = build_risk_system()

        if args.likelihood is None:
            likelihood = float(input("Likelihood (0-10): ").strip())
        else:
            likelihood = float(args.likelihood)

        if args.impact is None:
            impact = float(input("Impact (0-10): ").strip())
        else:
            impact = float(args.impact)

        out = system.infer({"likelihood": likelihood, "impact": impact})
        r = out["risk"]

        print("\n--- Fuzzy Risk Assessment ---")
        print(f"Inputs: likelihood={likelihood:.2f}, impact={impact:.2f}")
        print(f"Risk score (0-10): {r:.2f}  => level: {risk_label(r)}")
        return

    # Default: GUI
    run_gui()


if __name__ == "__main__":
    main()
