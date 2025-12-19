"""Fuzzy Student Grade Evaluator (standalone).

Run:
  python fuzzy_grade.py
  python fuzzy_grade.py --attendance 90 --assignments 75 --exam 80

Inputs:
- attendance: 0..100
- assignments: 0..100
- exam: 0..100
Output:
- fuzzy grade: 0..100 (also prints letter)

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


def build_grade_system() -> MamdaniSystem:
    attendance = FuzzyVariable(
        name="attendance",
        start=0.0,
        end=100.0,
        step=1.0,
        sets={
            "low": trapmf(0.0, 0.0, 40.0, 60.0),
            "medium": trimf(50.0, 70.0, 85.0),
            "high": trapmf(80.0, 90.0, 100.0, 100.0),
        },
    )

    assignments = FuzzyVariable(
        name="assignments",
        start=0.0,
        end=100.0,
        step=1.0,
        sets={
            "low": trapmf(0.0, 0.0, 40.0, 60.0),
            "medium": trimf(50.0, 70.0, 85.0),
            "high": trapmf(80.0, 90.0, 100.0, 100.0),
        },
    )

    exam = FuzzyVariable(
        name="exam",
        start=0.0,
        end=100.0,
        step=1.0,
        sets={
            "low": trapmf(0.0, 0.0, 35.0, 55.0),
            "medium": trimf(45.0, 65.0, 80.0),
            "high": trapmf(75.0, 88.0, 100.0, 100.0),
        },
    )

    grade = FuzzyVariable(
        name="grade",
        start=0.0,
        end=100.0,
        step=1.0,
        sets={
            "D": trapmf(0.0, 0.0, 45.0, 60.0),
            "C": trimf(55.0, 65.0, 75.0),
            "B": trimf(70.0, 80.0, 88.0),
            "A": trapmf(85.0, 92.0, 100.0, 100.0),
        },
    )

    rules: List[FuzzyRule] = [
        FuzzyRule([("exam", "low")], ("grade", "D")),
        FuzzyRule([("exam", "low"), ("assignments", "high")], ("grade", "C")),

        FuzzyRule([("exam", "high"), ("assignments", "high"), ("attendance", "high")], ("grade", "A")),
        FuzzyRule([("exam", "high"), ("assignments", "high")], ("grade", "A")),

        FuzzyRule([("exam", "high"), ("assignments", "medium")], ("grade", "B")),
        FuzzyRule([("exam", "medium"), ("assignments", "high")], ("grade", "B")),

        FuzzyRule([("exam", "medium"), ("assignments", "medium")], ("grade", "C")),
        FuzzyRule([("exam", "medium"), ("assignments", "low")], ("grade", "C")),

        FuzzyRule([("attendance", "low"), ("exam", "medium")], ("grade", "C")),
        FuzzyRule([("attendance", "low"), ("exam", "low")], ("grade", "D")),
    ]

    return MamdaniSystem(
        inputs={"attendance": attendance, "assignments": assignments, "exam": exam},
        outputs={"grade": grade},
        rules=rules,
    )


def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Fuzzy Student Grade Evaluator")
    p.add_argument("--attendance", type=float, help="Attendance percent 0..100")
    p.add_argument("--assignments", type=float, help="Assignments score 0..100")
    p.add_argument("--exam", type=float, help="Exam score 0..100")
    p.add_argument("--prompt", action="store_true", help="Prompt for inputs in terminal (no GUI)")
    return p


def run_gui() -> None:
    system = build_grade_system()

    root = tk.Tk()
    root.title("Fuzzy Student Grade Evaluator")
    root.resizable(False, False)

    container = ttk.Frame(root, padding=12)
    container.grid(row=0, column=0, sticky="nsew")

    ttk.Label(container, text="Fuzzy Student Grade Evaluator", font=("Segoe UI", 14, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w"
    )

    att_var = tk.DoubleVar(value=80.0)
    asg_var = tk.DoubleVar(value=80.0)
    exam_var = tk.DoubleVar(value=80.0)

    def row(label: str, var: tk.DoubleVar, r: int) -> ttk.Label:
        ttk.Label(container, text=label).grid(row=r, column=0, sticky="w", pady=(10, 0))
        ttk.Scale(container, from_=0.0, to=100.0, variable=var).grid(row=r, column=1, sticky="ew", padx=(10, 10), pady=(10, 0))
        value_lbl = ttk.Label(container, width=6)
        value_lbl.grid(row=r, column=2, sticky="e", pady=(10, 0))
        return value_lbl

    att_lbl = row("Attendance (0–100)", att_var, 1)
    asg_lbl = row("Assignments (0–100)", asg_var, 2)
    exam_lbl = row("Exam (0–100)", exam_var, 3)

    grade_var = tk.StringVar(value="—")
    letter_var = tk.StringVar(value="")

    out_box = ttk.Labelframe(container, text="Result", padding=10)
    out_box.grid(row=4, column=0, columnspan=3, sticky="ew", pady=(12, 0))
    ttk.Label(out_box, text="Fuzzy grade:").grid(row=0, column=0, sticky="w")
    ttk.Label(out_box, textvariable=grade_var, font=("Segoe UI", 16, "bold")).grid(row=0, column=1, sticky="e")
    ttk.Label(out_box, textvariable=letter_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def sync_labels() -> None:
        att_lbl.configure(text=f"{att_var.get():.0f}")
        asg_lbl.configure(text=f"{asg_var.get():.0f}")
        exam_lbl.configure(text=f"{exam_var.get():.0f}")

    def compute() -> None:
        attendance = float(att_var.get())
        assignments = float(asg_var.get())
        exam = float(exam_var.get())

        out = system.infer({"attendance": attendance, "assignments": assignments, "exam": exam})
        g = float(out["grade"])
        grade_var.set(f"{g:.1f}")
        letter_var.set(f"Letter: {grade_letter(g)}")

    btn_row = ttk.Frame(container)
    btn_row.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(10, 0))
    ttk.Button(btn_row, text="Compute", command=compute).pack(side=tk.LEFT)
    ttk.Button(btn_row, text="Close", command=root.destroy).pack(side=tk.RIGHT)

    container.columnconfigure(1, weight=1)

    att_var.trace_add("write", lambda *_: sync_labels())
    asg_var.trace_add("write", lambda *_: sync_labels())
    exam_var.trace_add("write", lambda *_: sync_labels())
    sync_labels()
    compute()

    root.mainloop()


def main() -> None:
    args = build_parser().parse_args()

    # If user passes any values or requests prompting, run CLI mode.
    if args.prompt or args.attendance is not None or args.assignments is not None or args.exam is not None:
        system = build_grade_system()

        if args.attendance is None:
            attendance = float(input("Attendance % (0-100): ").strip())
        else:
            attendance = float(args.attendance)

        if args.assignments is None:
            assignments = float(input("Assignments score (0-100): ").strip())
        else:
            assignments = float(args.assignments)

        if args.exam is None:
            exam = float(input("Exam score (0-100): ").strip())
        else:
            exam = float(args.exam)

        out = system.infer({"attendance": attendance, "assignments": assignments, "exam": exam})
        g = out["grade"]

        print("\n--- Fuzzy Student Grade Evaluator ---")
        print(f"Inputs: attendance={attendance:.1f}, assignments={assignments:.1f}, exam={exam:.1f}")
        print(f"Fuzzy grade (0-100): {g:.1f}  => letter: {grade_letter(g)}")
        return

    # Default: GUI
    run_gui()


if __name__ == "__main__":
    main()
