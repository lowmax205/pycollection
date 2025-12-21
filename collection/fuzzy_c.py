import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class GroundStabilityFuzzyController:
    def __init__(self):
        self.setup_membership_functions()
        self.setup_rule_base()

    def setup_membership_functions(self):
        # Membership functions from the photo
        # Lithology (input)
        self.lithology_mf = {
            'WF': (0, 1, 2),   # Weakly Fractured/Weathered
            'MF': (1.5, 3.25, 5), # Moderately Fractured/Weathered
            'HF': (4, 6, 8),   # Highly Fractured/Weathered
            'VF': (7, 8.5, 10) # Very Fractured/Weathered
        }
        # Ground Stability (input)
        self.stability_mf = {
            'ST': (0, 1, 2),   # Stable
            'SC': (1.5, 3.25, 5), # Presence of Soil Creep
            'IL': (4, 6, 8),   # Inactive landslide
            'AL': (7, 8.5, 10) # Active landslide evident
        }
        # Output (Ground Stability Fuzzy Variable)
        self.gs_mf = {
            'L': (0, 1, 2),    # Low
            'M': (1.5, 3.25, 5), # Moderate
            'H': (4, 6, 8),   # High
            'VH': (7, 8.5, 10) # Very High
        }

    def setup_rule_base(self):
        # Table 5 from the photo
        self.rule_table = {
            'WF': {'ST': 'L', 'SC': 'L', 'IL': 'M',  'AL': 'H'},
            'MF': {'ST': 'L', 'SC': 'M', 'IL': 'H',  'AL': 'VH'},
            'HF': {'ST': 'M', 'SC': 'H', 'IL': 'VH', 'AL': 'VH'},
            'VF': {'ST': 'VH', 'SC': 'VH', 'IL': 'VH', 'AL': 'VH'}
        }

    def triangular_mf_with_log(self, x, x0, x1, x2, label):
        mu = 0.0
        if x <= x0 or x >= x2:
            return 0.0, f"{label}: Out of range (0.00)"
        elif x == x1:
            return 1.0, f"{label}: Peak (1.00)"
        elif x0 < x < x1:
            mu = 0 + (x - x0) / (x1 - x0)
            return mu, f"{label} (Left):\n0 +({x:.2f}-{x0})/({x1-x0}) = {mu:.3f}"
        else:
            mu = 0 + (x2 - x) / (x2 - x1)
            return mu, f"{label} (Right):\n0 + ({x2}-{x:.2f})/({x2-x1}) = {mu:.3f}"

    def fuzzify_and_log(self, value, mf_dict):
        results, logs = {}, []
        for term, params in mf_dict.items():
            mu, log_msg = self.triangular_mf_with_log(value, *params, term)
            results[term] = mu
            if mu > 0:
                logs.append(log_msg)
        return results, logs

    def evaluate_rules(self, litho_fz, stability_fz):
        fired = []
        for l_term, l_mu in litho_fz.items():
            for s_term, s_mu in stability_fz.items():
                if l_mu > 0 and s_mu > 0:
                    strength = min(l_mu, s_mu)
                    out_term = self.rule_table[l_term][s_term]
                    log_entry = f"IF {l_term}({l_mu:.2f}) AND {s_term}({s_mu:.2f}) THEN {out_term} [{strength:.2f}]"
                    fired.append((strength, out_term, log_entry))
        return fired

    def compute_manual_cog_with_log(self, fired_rules):
        total_weighted_area = 0.0
        total_area = 0.0
        log_steps = []
        for strength, term, _ in fired_rules:
            a, b, c = self.gs_mf[term]
            base = (c - a)
            area = strength * base
            centroid = b
            weighted = area * centroid
            total_weighted_area += weighted
            total_area += area
            log_steps.append(f"{term}: Area({strength:.2f}*{base}) * Centroid({centroid}) = {weighted:.2f}")
        sum_wa = round(total_weighted_area, 2)
        sum_a = round(total_area, 2)
        cog = sum_wa / sum_a if sum_a != 0 else 0.0
        final_log = "\n".join(log_steps)
        final_log += f"\n\nΣ(Area*Centroid): {sum_wa:.2f}"
        final_log += f"\nΣ(Area): {sum_a:.2f}"
        final_log += f"\nCOG = {sum_wa:.2f} / {sum_a:.2f}"
        final_log += f"\nCOG = {cog:.2f}"
        return cog, final_log

    def aggregate_output(self, fired_rules, resolution=500):
        x = np.linspace(0, 10, resolution)
        agg = np.zeros(resolution)
        for strength, output_term, _ in fired_rules:
            p = self.gs_mf[output_term]
            mf_vals = np.array([self.simple_tri(xi, *p) for xi in x])
            agg = np.maximum(agg, np.minimum(mf_vals, strength))
        return x, agg

    def simple_tri(self, x, x0, x1, x2):
        if x <= x0 or x >= x2: return 0.0
        if x0 < x <= x1: return (x - x0) / (x1 - x0)
        return (x2 - x) / (x2 - x1)

class FuzzyControllerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Ground Stability - Fuzzy Logic Controller")
        self.root.geometry("1300x900")
        self.style = ttk.Style()
        try:
            self.style.theme_use("clam")
        except Exception:
            pass
        self.controller = GroundStabilityFuzzyController()
        self.create_gui()
        self.update_computation()

    def create_gui(self):
        # Main Paned layout: left controls, right notebook plots
        paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        left_frame = ttk.Frame(paned)
        right_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=1)
        paned.add(right_frame, weight=3)

        # Inputs Section
        input_frame = ttk.LabelFrame(left_frame, text="Inputs")
        input_frame.pack(fill=tk.X, padx=8, pady=8)

        self.litho_var = tk.DoubleVar(value=7.8)
        self.stab_var = tk.DoubleVar(value=7.5)
        self.realtime_var = tk.BooleanVar(value=True)

        # Lithology control
        row = ttk.Frame(input_frame)
        row.pack(fill=tk.X, pady=4)
        ttk.Label(row, text="Lithology").pack(side=tk.LEFT)
        ttk.Scale(row, from_=0, to=10, variable=self.litho_var, orient=tk.HORIZONTAL,
                  command=lambda e: self.on_change() if self.realtime_var.get() else None, length=220).pack(side=tk.LEFT, padx=8)
        self.litho_lbl = ttk.Label(row, text=f"{self.litho_var.get():.2f}")
        self.litho_lbl.pack(side=tk.LEFT)

        # Ground Stability control
        row2 = ttk.Frame(input_frame)
        row2.pack(fill=tk.X, pady=4)
        ttk.Label(row2, text="Ground Stability").pack(side=tk.LEFT)
        ttk.Scale(row2, from_=0, to=10, variable=self.stab_var, orient=tk.HORIZONTAL,
                  command=lambda e: self.on_change() if self.realtime_var.get() else None, length=220).pack(side=tk.LEFT, padx=8)
        self.stab_lbl = ttk.Label(row2, text=f"{self.stab_var.get():.2f}")
        self.stab_lbl.pack(side=tk.LEFT)

        # Controls row: realtime toggle, compute, reset, theme, export
        ctrl_row = ttk.Frame(input_frame)
        ctrl_row.pack(fill=tk.X, pady=8)
        ttk.Checkbutton(ctrl_row, text="Realtime", variable=self.realtime_var, command=self.on_change).pack(side=tk.LEFT)
        ttk.Button(ctrl_row, text="Compute", command=self.update_computation).pack(side=tk.LEFT, padx=6)
        ttk.Button(ctrl_row, text="Reset", command=self.reset_inputs).pack(side=tk.LEFT, padx=6)

        ttk.Label(ctrl_row, text="Theme:").pack(side=tk.LEFT, padx=(12, 4))
        self.theme_var = tk.StringVar(value=self.style.theme_use())
        themes = [t for t in self.style.theme_names()]
        self.theme_combo = ttk.Combobox(ctrl_row, textvariable=self.theme_var, values=themes, width=10, state="readonly")
        self.theme_combo.pack(side=tk.LEFT)
        self.theme_combo.bind("<<ComboboxSelected>>", self.on_theme_change)

        ttk.Button(ctrl_row, text="Export Output Plot", command=self.export_output_plot).pack(side=tk.RIGHT)

        # Active Rules Section
        rule_frame = ttk.LabelFrame(left_frame, text="Active Rules")
        rule_frame.pack(fill=tk.BOTH, expand=False, padx=8, pady=(0, 8))
        self.rules_text = scrolledtext.ScrolledText(rule_frame, width=60, height=6, font=("Consolas", 9))
        self.rules_text.pack(fill=tk.BOTH, expand=True)

        # Membership Calculation Section
        formula_frame = ttk.LabelFrame(left_frame, text="Membership Calculation Verification")
        formula_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.formula_text = scrolledtext.ScrolledText(formula_frame, width=60, height=10, font=("Consolas", 9))
        self.formula_text.pack(fill=tk.BOTH, expand=True)

        # COG Calculation Console
        cog_calc_frame = ttk.LabelFrame(left_frame, text="Center of Gravity (COG) Calculation")
        cog_calc_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self.cog_calc_text = scrolledtext.ScrolledText(cog_calc_frame, width=60, height=8, font=("Consolas", 9))
        self.cog_calc_text.pack(fill=tk.BOTH, expand=True)

        # Status Bar
        self.res_lbl = ttk.Label(left_frame, text="COG Result: 0.00", anchor=tk.W, font=("Segoe UI", 14, "bold"))
        self.res_lbl.pack(fill=tk.X, padx=8, pady=(0, 8))

        # Tabbed Plots Section
        self.tab_control = ttk.Notebook(right_frame)
        self.tab_control.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.figures = {}; self.canvases = {}
        for name in ["Lithology", "Stability", "GS Output"]:
            frame = ttk.Frame(self.tab_control)
            self.tab_control.add(frame, text=name)
            fig = Figure(figsize=(6, 4.5), dpi=100)
            canvas = FigureCanvasTkAgg(fig, master=frame)
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.figures[name] = fig
            self.canvases[name] = canvas

    def on_change(self, e=None):
        self.litho_lbl.config(text=f"{self.litho_var.get():.2f}")
        self.stab_lbl.config(text=f"{self.stab_var.get():.2f}")
        self.update_computation()

    def reset_inputs(self):
        self.litho_var.set(7.8)
        self.stab_var.set(7.5)
        self.on_change()

    def on_theme_change(self, event=None):
        try:
            self.style.theme_use(self.theme_var.get())
        except Exception:
            pass

    def update_computation(self):
        lv, sv = self.litho_var.get(), self.stab_var.get()
        l_fz, l_logs = self.controller.fuzzify_and_log(lv, self.controller.lithology_mf)
        s_fz, s_logs = self.controller.fuzzify_and_log(sv, self.controller.stability_mf)
        fired = self.controller.evaluate_rules(l_fz, s_fz)
        x_out, agg = self.controller.aggregate_output(fired)
        cog, cog_log = self.controller.compute_manual_cog_with_log(fired)
        # Update Text UI
        self.res_lbl.config(text=f"COG Result: {cog:.2f}")
        self.rules_text.delete(1.0, tk.END)
        for _, _, desc in fired: self.rules_text.insert(tk.END, desc + "\n")
        self.formula_text.delete(1.0, tk.END)
        self.formula_text.insert(tk.END, "--- LITHOLOGY ---\n")
        for log in l_logs: self.formula_text.insert(tk.END, log + "\n")
        self.formula_text.insert(tk.END, "\n--- STABILITY ---\n")
        for log in s_logs: self.formula_text.insert(tk.END, log + "\n")
        self.cog_calc_text.delete(1.0, tk.END)
        self.cog_calc_text.insert(tk.END, cog_log)
        self.plot_all(lv, sv, x_out, agg, cog)

    def plot_all(self, lv, sv, x, agg, cog):
        # Plot Input MFs
        for name, mf_dict, val in [("Lithology", self.controller.lithology_mf, lv), ("Stability", self.controller.stability_mf, sv)]:
            ax = self.figures[name].add_subplot(111); ax.clear()
            xr = np.linspace(0, 10, 200)
            for t, p in mf_dict.items():
                y = [self.controller.simple_tri(xi, *p) for xi in xr]
                ax.plot(xr, y, label=t)
            ax.axvline(val, color='red', linestyle='--')
            ax.set_xlabel("Crisp Input (0-10)")
            ax.set_ylabel("Fuzzy Membership Degree (0-1)")
            ax.legend(); self.canvases[name].draw()
        # Plot Output/COG
        ax = self.figures["GS Output"].add_subplot(111); ax.clear()
        xr = np.linspace(0, 10, 250)
        for term, params in self.controller.gs_mf.items():
            y_mf = [self.controller.simple_tri(xi, *params) for xi in xr]
            ax.plot(xr, y_mf, '-', alpha=0.3, label=term)
        ax.fill_between(x, agg, color='purple', alpha=0.4, label='Aggregated Result')
        ax.axvline(cog, color='red', linewidth=2, linestyle='--')
        ax.set_ylim(0, 1.1)
        ax.annotate(f'COG = {cog:.2f}', xy=(cog, 0), xytext=(cog+0.5, 0.3),
                    arrowprops=dict(arrowstyle='->', color='red'), color='darkred')
        ax.legend(loc='upper right', fontsize='small')
        self.canvases["GS Output"].draw()

    def export_output_plot(self):
        try:
            fig = self.figures.get("GS Output")
            if fig is None:
                return
            fig.savefig("gs_output.png", dpi=150, bbox_inches='tight')
        except Exception:
            pass

if __name__ == "__main__":
    root = tk.Tk()
    app = FuzzyControllerGUI(root)
    root.mainloop()
