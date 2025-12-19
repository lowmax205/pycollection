"""Fuzzy Tip Calculator (standalone).

Run:
  python fuzzy_tip.py
  python fuzzy_tip.py --service 8 --food 7 --bill 120

Inputs:
- service quality: 0..10
- food quality: 0..10
Output:
- tip percent: 0..30

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


def build_tip_system() -> MamdaniSystem:
    service = FuzzyVariable(
        name="service",
        start=0.0,
        end=10.0,
        step=0.1,
        sets={
            "poor": trapmf(0.0, 0.0, 2.0, 4.0),
            "ok": trimf(2.5, 5.0, 7.5),
            "great": trapmf(6.0, 8.0, 10.0, 10.0),
        },
    )

    food = FuzzyVariable(
        name="food",
        start=0.0,
        end=10.0,
        step=0.1,
        sets={
            "poor": trapmf(0.0, 0.0, 2.0, 4.0),
            "ok": trimf(2.5, 5.0, 7.5),
            "great": trapmf(6.0, 8.0, 10.0, 10.0),
        },
    )

    tip = FuzzyVariable(
        name="tip",
        start=0.0,
        end=30.0,
        step=0.1,
        sets={
            "low": trapmf(0.0, 0.0, 5.0, 12.0),
            "medium": trimf(10.0, 15.0, 20.0),
            "high": trapmf(18.0, 22.0, 30.0, 30.0),
        },
    )

    rules: List[FuzzyRule] = [
        FuzzyRule([("service", "poor")], ("tip", "low")),
        FuzzyRule([("food", "poor")], ("tip", "low")),
        FuzzyRule([("service", "great"), ("food", "great")], ("tip", "high")),
        FuzzyRule([("service", "great"), ("food", "ok")], ("tip", "high")),
        FuzzyRule([("service", "ok"), ("food", "great")], ("tip", "high")),
        FuzzyRule([("service", "ok"), ("food", "ok")], ("tip", "medium")),
    ]

    return MamdaniSystem(inputs={"service": service, "food": food}, outputs={"tip": tip}, rules=rules)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fuzzy Tip Calculator")
    p.add_argument("--service", type=float, help="Service quality 0..10")
    p.add_argument("--food", type=float, help="Food quality 0..10")
    p.add_argument("--bill", type=float, help="Bill amount (optional)")
    p.add_argument("--prompt", action="store_true", help="Prompt for inputs in terminal (no GUI)")
    return p


def run_gui() -> None:
    system = build_tip_system()

    root = tk.Tk()
    root.title("Fuzzy Tip Calculator")
    root.resizable(False, False)

    container = ttk.Frame(root, padding=12)
    container.grid(row=0, column=0, sticky="nsew")

    title = ttk.Label(container, text="Fuzzy Tip Calculator", font=("Segoe UI", 14, "bold"))
    title.grid(row=0, column=0, columnspan=3, sticky="w")

    service_var = tk.DoubleVar(value=5.0)
    food_var = tk.DoubleVar(value=5.0)
    bill_var = tk.StringVar(value="")

    ttk.Label(container, text="Service (0–10)").grid(row=1, column=0, sticky="w", pady=(10, 0))
    service_scale = ttk.Scale(container, from_=0.0, to=10.0, variable=service_var)
    service_scale.grid(row=1, column=1, sticky="ew", padx=(10, 10), pady=(10, 0))
    service_lbl = ttk.Label(container, width=6)
    service_lbl.grid(row=1, column=2, sticky="e", pady=(10, 0))

    ttk.Label(container, text="Food (0–10)").grid(row=2, column=0, sticky="w", pady=(10, 0))
    food_scale = ttk.Scale(container, from_=0.0, to=10.0, variable=food_var)
    food_scale.grid(row=2, column=1, sticky="ew", padx=(10, 10), pady=(10, 0))
    food_lbl = ttk.Label(container, width=6)
    food_lbl.grid(row=2, column=2, sticky="e", pady=(10, 0))

    ttk.Label(container, text="Bill (optional)").grid(row=3, column=0, sticky="w", pady=(10, 0))
    bill_entry = ttk.Entry(container, textvariable=bill_var)
    bill_entry.grid(row=3, column=1, sticky="ew", padx=(10, 10), pady=(10, 0))
    ttk.Label(container, text="").grid(row=3, column=2, sticky="e", pady=(10, 0))

    tip_pct_var = tk.StringVar(value="—")
    tip_amt_var = tk.StringVar(value="")

    out_box = ttk.Labelframe(container, text="Result", padding=10)
    out_box.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))
    ttk.Label(out_box, text="Suggested tip:").grid(row=0, column=0, sticky="w")
    ttk.Label(out_box, textvariable=tip_pct_var, font=("Segoe UI", 16, "bold")).grid(row=0, column=1, sticky="e")
    ttk.Label(out_box, textvariable=tip_amt_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    err_var = tk.StringVar(value="")
    err_lbl = ttk.Label(container, textvariable=err_var, foreground="red")
    err_lbl.grid(row=5, column=0, columnspan=3, sticky="w", pady=(8, 0))

    def sync_labels() -> None:
        service_lbl.configure(text=f"{service_var.get():.1f}")
        food_lbl.configure(text=f"{food_var.get():.1f}")

    def compute() -> None:
        err_var.set("")
        tip_amt_var.set("")

        service = float(service_var.get())
        food = float(food_var.get())
        out = system.infer({"service": service, "food": food})
        tip_pct = float(out["tip"])
        tip_pct_var.set(f"{tip_pct:.2f}%")

        bill_txt = bill_var.get().strip()
        if bill_txt:
            try:
                bill = float(bill_txt)
            except ValueError:
                err_var.set("Bill must be a number (e.g., 120 or 120.50).")
                return
            tip_amt_var.set(f"Tip amount: {bill * tip_pct / 100.0:.2f}")

    btn_row = ttk.Frame(container)
    btn_row.grid(row=6, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    ttk.Button(btn_row, text="Compute", command=compute).pack(side=tk.LEFT)
    ttk.Button(btn_row, text="Close", command=root.destroy).pack(side=tk.RIGHT)

    container.columnconfigure(1, weight=1)

    service_var.trace_add("write", lambda *_: sync_labels())
    food_var.trace_add("write", lambda *_: sync_labels())
    sync_labels()
    compute()

    bill_entry.focus_set()
    root.mainloop()


def main() -> None:
    args = build_parser().parse_args()

    # If user passes any values or requests prompting, run CLI mode.
    if args.prompt or args.service is not None or args.food is not None or args.bill is not None:
        system = build_tip_system()

        if args.service is None:
            service = float(input("Service quality (0-10): ").strip())
        else:
            service = float(args.service)

        if args.food is None:
            food = float(input("Food quality (0-10): ").strip())
        else:
            food = float(args.food)

        bill = float(args.bill) if args.bill is not None else None

        out = system.infer({"service": service, "food": food})
        tip_pct = out["tip"]

        print("\n--- Fuzzy Tip Calculator ---")
        print(f"Inputs: service={service:.2f}, food={food:.2f}")
        print(f"Suggested tip: {tip_pct:.2f}%")
        if bill is not None:
            print(f"Tip amount (bill={bill:.2f}): {(bill * tip_pct / 100.0):.2f}")
        return

    # Default: GUI
    run_gui()


if __name__ == "__main__":
    main()
