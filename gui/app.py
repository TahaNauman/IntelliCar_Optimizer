# ─────────────────────────────────────────────
#  gui/app.py  –  Tkinter GUI
# ─────────────────────────────────────────────

import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from core.csp import generate_valid_configs, is_valid
from core.genetic import run_ga, pick_diverse_top
from core.fitness import evaluate_fitness, explain_fitness
from core.data import VARIABLES, CONSTRAINTS


# ── Color Palette ──────────────────────────────────────────────────────────────
BG        = "#1a1a2e"
PANEL     = "#16213e"
ACCENT    = "#0f3460"
HIGHLIGHT = "#e94560"
TEXT      = "#eaeaea"
SUBTEXT   = "#a0a0b0"
SUCCESS   = "#4ecca3"
WARN      = "#f5a623"
FONT_H    = ("Segoe UI", 13, "bold")
FONT_B    = ("Segoe UI", 10)
FONT_S    = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 9)


class CarConfigApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🚗 AI Car Configurator — CSP + Genetic Algorithm")
        self.geometry("1280x800")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._population     = []
        self._history        = []
        self._top_configs    = []
        self._is_running     = False
        self._stop_flag      = False  # Signal to kill background thread

        self.protocol("WM_DELETE_WINDOW", self._on_close)  # Handle X button

        self._build_ui()

    # ── UI Builder ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Top header bar
        header = tk.Frame(self, bg=ACCENT, pady=10)
        header.pack(fill="x")
        tk.Label(header, text="🚗  AI Car Configuration System",
                 font=("Segoe UI", 16, "bold"), bg=ACCENT, fg=TEXT).pack(side="left", padx=20)
        tk.Label(header, text="CSP  ×  Genetic Algorithm",
                 font=("Segoe UI", 10), bg=ACCENT, fg=SUBTEXT).pack(side="left")

        # Main body: left panel + right content
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=10, pady=10)

        self._build_left_panel(body)
        self._build_right_panel(body)

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=PANEL, bd=0, relief="flat", width=280)
        left.pack(side="left", fill="y", padx=(0, 10))
        left.pack_propagate(False)

        # Title
        tk.Label(left, text="⚙️  Preferences", font=FONT_H,
                 bg=PANEL, fg=TEXT).pack(pady=(18, 5), padx=15, anchor="w")
        tk.Label(left, text="Drag sliders to set your priorities",
                 font=FONT_S, bg=PANEL, fg=SUBTEXT).pack(padx=15, anchor="w")

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=15, pady=10)

        # Weight sliders
        self._sliders = {}
        slider_defs = [
            ("performance",  "🔥 Performance",  HIGHLIGHT),
            ("stability",    "🛡️  Stability",    SUCCESS),
            ("efficiency",   "⚡ Efficiency",   "#5bc0eb"),
            ("aerodynamics", "✈️  Aerodynamics", WARN),
        ]
        for key, label, color in slider_defs:
            self._add_slider(left, key, label, color)

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=15, pady=10)

        # Generations setting
        tk.Label(left, text="🔁 Generations", font=FONT_B, bg=PANEL, fg=TEXT).pack(padx=15, anchor="w")
        self._gen_var = tk.IntVar(value=30)
        gen_frame = tk.Frame(left, bg=PANEL)
        gen_frame.pack(fill="x", padx=15, pady=4)
        tk.Scale(gen_frame, from_=10, to=100, orient="horizontal",
                 variable=self._gen_var, bg=PANEL, fg=TEXT,
                 troughcolor=ACCENT, highlightthickness=0,
                 activebackground=HIGHLIGHT).pack(fill="x")
        self._gen_label = tk.Label(gen_frame, textvariable=self._gen_var,
                                   font=FONT_S, bg=PANEL, fg=SUBTEXT)
        self._gen_label.pack(anchor="e")

        ttk.Separator(left, orient="horizontal").pack(fill="x", padx=15, pady=10)

        # Run button
        self._run_btn = tk.Button(
            left, text="▶  Run GA", font=("Segoe UI", 11, "bold"),
            bg=HIGHLIGHT, fg="white", relief="flat", cursor="hand2",
            pady=10, command=self._start_ga
        )
        self._run_btn.pack(fill="x", padx=15, pady=5)

        # Status label
        self._status_var = tk.StringVar(value="Ready. Press Run to start.")
        tk.Label(left, textvariable=self._status_var,
                 font=FONT_S, bg=PANEL, fg=SUBTEXT,
                 wraplength=240).pack(padx=15, pady=6)

        # Progress bar
        self._progress = ttk.Progressbar(left, mode="determinate", maximum=100)
        self._progress.pack(fill="x", padx=15, pady=4)

    def _add_slider(self, parent, key, label, color):
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="x", padx=15, pady=4)
        tk.Label(frame, text=label, font=FONT_B, bg=PANEL, fg=color).pack(anchor="w")
        var = tk.DoubleVar(value=0.5)
        self._sliders[key] = var
        tk.Scale(frame, from_=0.0, to=1.0, resolution=0.05,
                 orient="horizontal", variable=var,
                 bg=PANEL, fg=TEXT, troughcolor=ACCENT,
                 highlightthickness=0, activebackground=color).pack(fill="x")

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True)

        # Notebook tabs
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook",        background=BG,    borderwidth=0)
        style.configure("TNotebook.Tab",    background=ACCENT, foreground=TEXT,
                        padding=[12, 6], font=FONT_B)
        style.map("TNotebook.Tab",          background=[("selected", HIGHLIGHT)])

        self._nb = ttk.Notebook(right)
        nb = self._nb
        nb.pack(fill="both", expand=True)

        # Tab 1 – Results
        self._tab_results = tk.Frame(nb, bg=BG)
        nb.add(self._tab_results, text="🏆  Top Configurations")
        self._build_results_tab(self._tab_results)

        # Tab 2 – Evolution Chart
        self._tab_chart = tk.Frame(nb, bg=BG)
        nb.add(self._tab_chart, text="📈  Evolution Chart")
        self._build_chart_tab(self._tab_chart)

        # Tab 3 – Constraint Validator
        self._tab_csp = tk.Frame(nb, bg=BG)
        nb.add(self._tab_csp, text="🔒  Constraint Validator")
        self._build_csp_tab(self._tab_csp)

        # Tab 4 – Reasoning
        self._tab_reason = tk.Frame(nb, bg=BG)
        nb.add(self._tab_reason, text="💡  Reasoning")
        self._build_reason_tab(self._tab_reason)

    # ── Tab: Results ──────────────────────────────────────────────────────────

    def _build_results_tab(self, parent):
        tk.Label(parent, text="Top 5 Configurations after GA Evolution",
                 font=FONT_H, bg=BG, fg=TEXT).pack(pady=(12, 4), padx=15, anchor="w")
        tk.Label(parent, text="Sorted by fitness score (highest first)",
                 font=FONT_S, bg=BG, fg=SUBTEXT).pack(padx=15, anchor="w")

        self._results_frame = tk.Frame(parent, bg=BG)
        self._results_frame.pack(fill="both", expand=True, padx=15, pady=10)

        tk.Label(self._results_frame, text="Run the GA to see results here.",
                 font=FONT_B, bg=BG, fg=SUBTEXT).pack(pady=40)

    def _render_results(self, top_configs, weights):
        for w in self._results_frame.winfo_children():
            w.destroy()

        cols = ["#", "Engine", "Tires", "Rims", "Suspension", "Transmission", "Body", "Turbo", "Fitness"]
        col_w = [30, 80, 80, 70, 100, 110, 70, 60, 70]

        # Header row
        hrow = tk.Frame(self._results_frame, bg=ACCENT)
        hrow.pack(fill="x", pady=(0, 2))
        for col, w in zip(cols, col_w):
            tk.Label(hrow, text=col, font=("Segoe UI", 9, "bold"),
                     bg=ACCENT, fg=TEXT, width=w//7, anchor="center").pack(side="left", padx=2)

        # Data rows
        for i, config in enumerate(top_configs[:5], 1):
            score = evaluate_fitness(config, weights)
            row_bg = PANEL if i % 2 == 0 else BG
            row = tk.Frame(self._results_frame, bg=row_bg, pady=6)
            row.pack(fill="x", pady=1)

            vals = [str(i), config["engine"], config["tires"], config["rims"],
                    config["suspension"], config["transmission"],
                    config["body_type"], config["turbo"], f"{score:.2f}"]
            for val, w in zip(vals, col_w):
                color = SUCCESS if val == vals[-1] else TEXT
                tk.Label(row, text=val, font=FONT_MONO, bg=row_bg, fg=color,
                         width=w//7, anchor="center").pack(side="left", padx=2)

            # Hover highlight + click to view reasoning
            def on_enter(e, f=row, bg=row_bg):
                f.configure(bg=ACCENT)
                for ch in f.winfo_children():
                    ch.configure(bg=ACCENT)

            def on_leave(e, f=row, bg=row_bg):
                f.configure(bg=bg)
                for ch in f.winfo_children():
                    ch.configure(bg=bg)

            def on_click(e, c=config):
                self._show_reasoning(c, self._get_weights())
                self._nb.select(self._tab_reason)

            row.bind("<Enter>",    on_enter)
            row.bind("<Leave>",    on_leave)
            row.bind("<Button-1>", on_click)
            for child in row.winfo_children():
                child.bind("<Enter>",    on_enter)
                child.bind("<Leave>",    on_leave)
                child.bind("<Button-1>", on_click)

        tk.Label(self._results_frame,
                 text="💡 Click any row — it will jump to the Reasoning tab automatically",
                 font=FONT_S, bg=BG, fg=SUBTEXT).pack(pady=8)

    # ── Tab: Evolution Chart ──────────────────────────────────────────────────

    def _build_chart_tab(self, parent):
        self._fig, self._ax = plt.subplots(figsize=(8, 4.5))
        self._fig.patch.set_facecolor("#1a1a2e")
        self._ax.set_facecolor("#16213e")
        self._ax.set_title("Fitness Score Over Generations", color=TEXT, fontsize=12)
        self._ax.set_xlabel("Generation", color=SUBTEXT)
        self._ax.set_ylabel("Best Fitness", color=SUBTEXT)
        self._ax.tick_params(colors=SUBTEXT)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(ACCENT)

        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _update_chart(self, history, mutation_history):
        self._ax.clear()
        self._ax.set_facecolor(PANEL)

        gens = range(1, len(history) + 1)

        # Primary axis — fitness
        self._ax.plot(gens, history, color=HIGHLIGHT, linewidth=2,
                      marker="o", markersize=3, label="Best Fitness")
        self._ax.fill_between(gens, history, alpha=0.12, color=HIGHLIGHT)
        self._ax.set_ylabel("Best Fitness", color=HIGHLIGHT)
        self._ax.tick_params(axis="y", colors=HIGHLIGHT)
        self._ax.set_xlabel("Generation", color=SUBTEXT)
        self._ax.tick_params(axis="x", colors=SUBTEXT)

        # Secondary axis — adaptive mutation rate
        ax2 = self._ax.twinx()
        ax2.plot(gens, mutation_history, color=WARN, linewidth=1.5,
                 linestyle="--", alpha=0.8, label="Mutation Rate")
        ax2.set_ylabel("Mutation Rate", color=WARN)
        ax2.tick_params(axis="y", colors=WARN)
        ax2.set_ylim(0, 0.7)
        ax2.set_facecolor(PANEL)

        self._ax.set_title("Fitness Evolution + Adaptive Mutation Rate",
                            color=TEXT, fontsize=11)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(ACCENT)

        # Combined legend
        lines1, labels1 = self._ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self._ax.legend(lines1 + lines2, labels1 + labels2,
                        loc="lower right", facecolor=PANEL, labelcolor=TEXT,
                        fontsize=8)
        self._canvas.draw()

    # ── Tab: CSP Constraint Validator ─────────────────────────────────────────

    def _build_csp_tab(self, parent):
        tk.Label(parent, text="Constraint Validation for Top Config",
                 font=FONT_H, bg=BG, fg=TEXT).pack(pady=(12, 4), padx=15, anchor="w")

        self._csp_frame = tk.Frame(parent, bg=BG)
        self._csp_frame.pack(fill="both", expand=True, padx=15, pady=10)

        tk.Label(self._csp_frame, text="Run the GA to validate constraints here.",
                 font=FONT_B, bg=BG, fg=SUBTEXT).pack(pady=40)

    def _render_csp(self, config):
        for w in self._csp_frame.winfo_children():
            w.destroy()

        rule_labels = [
            "Electric engine → No transmission",
            "Turbo only with VTEC or V8 (not Electric)",
            "Servis tires → Alloy or Steel rims only",
            "V8 engine → Sport or Heavy-Duty suspension",
            "Turbo → Sport or Heavy-Duty suspension",
            "Carbon rims → Sport or Heavy-Duty suspension",
            "Electric → No turbo (explicit check)",
        ]

        tk.Label(self._csp_frame,
                 text=f"Validating: {config['engine']} / {config['tires']} / {config['turbo']} turbo",
                 font=FONT_B, bg=BG, fg=SUBTEXT).pack(anchor="w", pady=(0, 8))

        for i, (constraint, label) in enumerate(zip(CONSTRAINTS, rule_labels)):
            satisfied = constraint(config)
            icon  = "✅" if satisfied else "❌"
            color = SUCCESS if satisfied else HIGHLIGHT
            row = tk.Frame(self._csp_frame, bg=PANEL, pady=6)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"  {icon}  Rule {i+1}: {label}",
                     font=FONT_B, bg=PANEL, fg=color).pack(anchor="w", padx=10)

    # ── Tab: Reasoning ────────────────────────────────────────────────────────

    def _build_reason_tab(self, parent):
        tk.Label(parent, text="Configuration Reasoning",
                 font=FONT_H, bg=BG, fg=TEXT).pack(pady=(12, 4), padx=15, anchor="w")
        tk.Label(parent, text="Click any result row to load reasoning here",
                 font=FONT_S, bg=BG, fg=SUBTEXT).pack(padx=15, anchor="w")

        self._reason_text = scrolledtext.ScrolledText(
            parent, font=FONT_MONO, bg=PANEL, fg=TEXT,
            insertbackground=TEXT, relief="flat",
            padx=14, pady=14, wrap="word"
        )
        self._reason_text.pack(fill="both", expand=True, padx=15, pady=10)
        self._reason_text.insert("end", "Run the GA and click a result row to see reasoning.")
        self._reason_text.configure(state="disabled")

    def _show_reasoning(self, config, weights):
        explanation = explain_fitness(config, weights)
        self._reason_text.configure(state="normal")
        self._reason_text.delete("1.0", "end")
        self._reason_text.insert("end", explanation)
        self._reason_text.configure(state="disabled")
        self._render_csp(config)

    def _on_close(self):
        """Called when user clicks the X button. Signals background thread to stop."""
        self._stop_flag = True
        self.destroy()
        sys.exit(0)

    # ── GA Runner ─────────────────────────────────────────────────────────────

    def _get_weights(self):
        return {k: v.get() for k, v in self._sliders.items()}

    def _start_ga(self):
        if self._is_running:
            return
        self._is_running = True
        self._run_btn.configure(state="disabled", text="⏳ Running...")
        self._status_var.set("Step 1/3: Generating valid configs via CSP...")
        self._progress["value"] = 0
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        weights = self._get_weights()
        generations = self._gen_var.get()

        # Step 1: CSP
        self.after(0, lambda: self._status_var.set("Generating valid configs via CSP..."))
        population = generate_valid_configs(80)
        if self._stop_flag:
            return
        self.after(0, lambda: self._progress.configure(value=20))

        # Step 2: GA
        self.after(0, lambda: self._status_var.set(f"Running GA for {generations} generations..."))

        def progress_cb(gen, fitness):
            if self._stop_flag:
                return
            pct = 20 + int((gen / generations) * 75)
            self.after(0, lambda p=pct, g=gen, f=fitness: (
                self._progress.configure(value=p),
                self._status_var.set(f"Generation {g}/{generations} — Best: {f:.2f}")
            ))

        final_pop, history, mutation_history = run_ga(
            population, weights, generations=generations,
            progress_callback=progress_cb,
            stop_flag=lambda: self._stop_flag
        )
        if self._stop_flag:
            return

        # Step 3: Update UI
        self.after(0, lambda: self._finish(final_pop, history, mutation_history, weights))

    def _finish(self, final_pop, history, mutation_history, weights):
        self._top_configs = pick_diverse_top(final_pop, weights, n=5, min_diff=2)
        self._history = history

        self._render_results(self._top_configs, weights)
        self._update_chart(history, mutation_history)

        best = self._top_configs[0]
        self._render_csp(best)
        self._show_reasoning(best, weights)

        self._progress["value"] = 100
        self._status_var.set(f"✅ Done! Best fitness: {evaluate_fitness(best, weights):.2f}")
        self._run_btn.configure(state="normal", text="▶  Run GA")
        self._is_running = False


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CarConfigApp()
    app.mainloop()