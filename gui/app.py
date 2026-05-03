# ─────────────────────────────────────────────
#  gui/app.py  –  Tkinter GUI (improved)
# ─────────────────────────────────────────────

import tkinter as tk
from tkinter import ttk
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
from core.fitness import evaluate_fitness, explain_fitness, _aggregate_stats
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
BLUE      = "#5bc0eb"
CARD_BG   = "#1e2a45"

FONT_H    = ("Segoe UI", 13, "bold")
FONT_B    = ("Segoe UI", 10)
FONT_S    = ("Segoe UI", 9)
FONT_MONO = ("Consolas", 10)

RANK_COLORS  = [WARN, "#c0c0c0", "#cd7f32", "#a78bfa", "#60a5fa"]
RANK_LABELS  = ["🥇", "🥈", "🥉", "🎖️", "🎖️"]
COMPONENT_ICONS = {
    "engine": "⚙️", "tires": "🔄", "rims": "⭕",
    "suspension": "🔩", "transmission": "🔧", "body_type": "🚘", "turbo": "💨",
}


class CarConfigApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🚗 AI Car Configurator — CSP + Genetic Algorithm")
        self.geometry("1360x860")
        self.configure(bg=BG)
        self.resizable(True, True)

        self._population     = []
        self._history        = []
        self._top_configs    = []
        self._is_running     = False
        self._stop_flag      = False
        self._slider_labels  = {}   # key → val_label widget
        self._card_widgets   = []   # list of {card, inner, top_row, chip_row, config}
        self._selected_index = None

        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()

    # ── UI Builder ─────────────────────────────────────────────────────────────

    def _build_ui(self):
        self._build_header()
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self._build_left_panel(body)
        self._build_right_panel(body)

    # ── Header ─────────────────────────────────────────────────────────────────

    def _build_header(self):
        header = tk.Frame(self, bg=ACCENT)
        header.pack(fill="x")

        left_hdr = tk.Frame(header, bg=ACCENT)
        left_hdr.pack(side="left", padx=20, pady=10)
        tk.Label(left_hdr, text="🚗  AI Car Configuration System",
                 font=("Segoe UI", 15, "bold"), bg=ACCENT, fg=TEXT).pack(anchor="w")
        tk.Label(left_hdr, text="Constraint Satisfaction Problem  ×  Genetic Algorithm",
                 font=("Segoe UI", 9), bg=ACCENT, fg=SUBTEXT).pack(anchor="w")

        self._stat_frame = tk.Frame(header, bg=ACCENT)
        self._stat_frame.pack(side="right", padx=20)
        self._stat_vars = {}
        for label, key in [("Best Fitness","fitness"),("Generations","gens"),
                            ("Valid Configs","configs"),("Improvement","improve")]:
            col = tk.Frame(self._stat_frame, bg=ACCENT)
            col.pack(side="left", padx=18, pady=8)
            var = tk.StringVar(value="—")
            self._stat_vars[key] = var
            tk.Label(col, textvariable=var, font=("Segoe UI", 16, "bold"),
                     bg=ACCENT, fg=HIGHLIGHT).pack()
            tk.Label(col, text=label, font=("Segoe UI", 8),
                     bg=ACCENT, fg=SUBTEXT).pack()

    def _update_stats(self, fitness, gens, configs, improve):
        # Animate numbers counting up over ~600ms
        steps = 20
        delay = 30  # ms per step
        targets = {
            "fitness": fitness,
            "gens":    float(gens),
            "configs": float(configs),
            "improve": improve,
        }
        def _step(n):
            if n > steps:
                return
            frac = n / steps
            self._stat_vars["fitness"].set(f"{fitness * frac:.1f}")
            self._stat_vars["gens"].set(str(int(gens * frac)))
            self._stat_vars["configs"].set(str(int(configs * frac)))
            self._stat_vars["improve"].set(f"+{improve * frac:.1f}")
            if n < steps:
                self.after(delay, lambda: _step(n + 1))
        _step(1)

    # ── Left Panel ─────────────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        left = tk.Frame(parent, bg=PANEL, width=290)
        left.pack(side="left", fill="y", padx=(0, 12), pady=(12, 0))
        left.pack_propagate(False)

        self._section(left, "⚙️  Preferences", "Set your optimization priorities")

        self._sliders = {}
        for key, label, color in [
            ("performance",  "🔥 Performance",  HIGHLIGHT),
            ("stability",    "🛡️  Stability",    SUCCESS),
            ("efficiency",   "⚡ Efficiency",   BLUE),
            ("aerodynamics", "✈️  Aerodynamics", WARN),
        ]:
            self._add_slider(left, key, label, color)

        self._divider(left)
        self._section(left, "🔁  Generations", None)

        self._gen_var = tk.IntVar(value=30)
        gen_frame = tk.Frame(left, bg=PANEL)
        gen_frame.pack(fill="x", padx=15, pady=(0, 4))
        gen_row = tk.Frame(gen_frame, bg=PANEL)
        gen_row.pack(fill="x")
        self._gen_disp = tk.Label(gen_row, text="30 gens", font=FONT_S, bg=PANEL, fg=WARN, width=8)
        self._gen_disp.pack(side="right")
        tk.Scale(gen_row, from_=10, to=100, orient="horizontal",
                 variable=self._gen_var, bg=PANEL, fg=TEXT,
                 troughcolor=ACCENT, highlightthickness=0,
                 activebackground=HIGHLIGHT, showvalue=False,
                 command=lambda v: self._gen_disp.config(text=f"{int(float(v))} gens")
                 ).pack(side="left", fill="x", expand=True)

        self._divider(left)

        self._run_btn = tk.Button(
            left, text="▶   Run GA", font=("Segoe UI", 11, "bold"),
            bg=HIGHLIGHT, fg="white", relief="flat", cursor="hand2",
            pady=12, command=self._start_ga,
            activebackground="#c73652", activeforeground="white"
        )
        self._run_btn.pack(fill="x", padx=15, pady=(8, 4))

        tk.Button(
            left, text="↺  Reset Sliders", font=("Segoe UI", 9),
            bg=ACCENT, fg=SUBTEXT, relief="flat", cursor="hand2",
            pady=6, command=self._reset_sliders,
            activebackground="#1a3a6e", activeforeground=TEXT
        ).pack(fill="x", padx=15, pady=(0, 8))

        self._status_var = tk.StringVar(value="Ready — configure and press Run")
        tk.Label(left, textvariable=self._status_var, font=FONT_S,
                 bg=PANEL, fg=SUBTEXT, wraplength=250, justify="left").pack(padx=15, anchor="w")

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Red.Horizontal.TProgressbar",
                        troughcolor=ACCENT, background=HIGHLIGHT, borderwidth=0)
        self._progress = ttk.Progressbar(left, mode="determinate", maximum=100,
                                          style="Red.Horizontal.TProgressbar")
        self._progress.pack(fill="x", padx=15, pady=8)

    def _section(self, parent, title, subtitle):
        tk.Label(parent, text=title, font=FONT_H,
                 bg=PANEL, fg=TEXT).pack(pady=(14, 2), padx=15, anchor="w")
        if subtitle:
            tk.Label(parent, text=subtitle, font=FONT_S,
                     bg=PANEL, fg=SUBTEXT).pack(padx=15, anchor="w", pady=(0, 4))

    def _divider(self, parent):
        ttk.Separator(parent, orient="horizontal").pack(fill="x", padx=15, pady=10)

    def _add_slider(self, parent, key, label, color):
        frame = tk.Frame(parent, bg=PANEL)
        frame.pack(fill="x", padx=15, pady=3)
        header_row = tk.Frame(frame, bg=PANEL)
        header_row.pack(fill="x")
        tk.Label(header_row, text=label, font=FONT_B, bg=PANEL, fg=color).pack(side="left")
        val_label = tk.Label(header_row, text="0.50", font=("Segoe UI", 9, "bold"),
                              bg=PANEL, fg=color)
        val_label.pack(side="right")
        var = tk.DoubleVar(value=0.5)
        self._sliders[key] = var
        self._slider_labels[key] = val_label   # store ref for reset
        tk.Scale(frame, from_=0.0, to=1.0, resolution=0.05,
                 orient="horizontal", variable=var,
                 command=lambda v, lbl=val_label: lbl.config(text=f"{float(v):.2f}"),
                 bg=PANEL, fg=TEXT, troughcolor=ACCENT,
                 highlightthickness=0, activebackground=color,
                 showvalue=False).pack(fill="x")

    # ── Right Panel ────────────────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        right = tk.Frame(parent, bg=BG)
        right.pack(side="left", fill="both", expand=True, pady=(12, 0))

        style = ttk.Style()
        style.configure("TNotebook",     background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=ACCENT, foreground=TEXT,
                        padding=[14, 7], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",       background=[("selected", HIGHLIGHT)])

        self._nb = ttk.Notebook(right)
        self._nb.pack(fill="both", expand=True)

        self._tab_results = tk.Frame(self._nb, bg=BG)
        self._nb.add(self._tab_results, text="🏆  Top Configurations")
        self._build_results_tab(self._tab_results)

        self._tab_chart = tk.Frame(self._nb, bg=BG)
        self._nb.add(self._tab_chart, text="📈  Evolution")
        self._build_chart_tab(self._tab_chart)

        self._tab_csp = tk.Frame(self._nb, bg=BG)
        self._nb.add(self._tab_csp, text="🔒  Constraints")
        self._build_csp_tab(self._tab_csp)

        self._tab_reason = tk.Frame(self._nb, bg=BG)
        self._nb.add(self._tab_reason, text="💡  Reasoning")
        self._build_reason_tab(self._tab_reason)

    # ── Tab: Results ──────────────────────────────────────────────────────────

    def _build_results_tab(self, parent):
        top_bar = tk.Frame(parent, bg=BG)
        top_bar.pack(fill="x", padx=15, pady=(14, 6))
        tk.Label(top_bar, text="Top 5 Configurations",
                 font=FONT_H, bg=BG, fg=TEXT).pack(side="left")
        tk.Label(top_bar, text="Click a card to see full reasoning →",
                 font=FONT_S, bg=BG, fg=SUBTEXT).pack(side="right")

        # Scrollable canvas wrapper
        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill="both", expand=True, padx=15, pady=(0, 10))

        self._results_canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical",
                                  command=self._results_canvas.yview)
        self._results_canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        self._results_canvas.pack(side="left", fill="both", expand=True)

        self._results_frame = tk.Frame(self._results_canvas, bg=BG)
        self._results_window = self._results_canvas.create_window(
            (0, 0), window=self._results_frame, anchor="nw"
        )

        def _on_frame_configure(e):
            self._results_canvas.configure(
                scrollregion=self._results_canvas.bbox("all"))

        def _on_canvas_configure(e):
            self._results_canvas.itemconfig(
                self._results_window, width=e.width)

        self._results_frame.bind("<Configure>", _on_frame_configure)
        self._results_canvas.bind("<Configure>", _on_canvas_configure)
        self._results_canvas.bind_all("<MouseWheel>",
            lambda e: self._results_canvas.yview_scroll(-1*(e.delta//120), "units"))

        self._empty_state(self._results_frame, "🧬", "Run the GA to see results",
                          "Configure your priorities and press Run GA")
        self._selected_index = None  # Track currently selected card

    def _empty_state(self, parent, icon, title, subtitle):
        f = tk.Frame(parent, bg=BG)
        f.pack(expand=True, pady=60)
        tk.Label(f, text=icon, font=("Segoe UI", 36), bg=BG, fg=SUBTEXT).pack()
        tk.Label(f, text=title, font=("Segoe UI", 13, "bold"),
                 bg=BG, fg=SUBTEXT).pack(pady=(8, 2))
        tk.Label(f, text=subtitle, font=FONT_S,
                 bg=BG, fg="#606070").pack()

    def _render_results(self, top_configs, weights):
        for w in self._results_frame.winfo_children():
            w.destroy()
        self._card_widgets = []
        self._selected_index = None

        scores    = [evaluate_fitness(c, weights) for c in top_configs]
        max_score = max(scores) if scores else 1

        for i, config in enumerate(top_configs[:5]):
            score    = scores[i]
            rank_col = RANK_COLORS[i]
            rank_lbl = RANK_LABELS[i]
            pct      = score / max_score if max_score else 0

            card = tk.Frame(self._results_frame, bg=CARD_BG)
            card.pack(fill="x", pady=5)

            # Left color accent bar
            tk.Frame(card, bg=rank_col, width=5).pack(side="left", fill="y")

            inner = tk.Frame(card, bg=CARD_BG, padx=12, pady=10)
            inner.pack(side="left", fill="both", expand=True)

            top_row = tk.Frame(inner, bg=CARD_BG)
            top_row.pack(fill="x")

            tk.Label(top_row, text=rank_lbl, font=("Segoe UI", 18),
                     bg=CARD_BG, fg=rank_col).pack(side="left", padx=(0, 10))

            chip_row = tk.Frame(top_row, bg=CARD_BG)
            chip_row.pack(side="left", fill="x", expand=True)
            for var in VARIABLES:
                chip = tk.Frame(chip_row, bg=ACCENT, padx=6, pady=2)
                chip.pack(side="left", padx=3, pady=2)
                tk.Label(chip, text=f"{COMPONENT_ICONS.get(var,'')} {config[var]}",
                         font=("Segoe UI", 11), bg=ACCENT, fg=TEXT).pack()

            tk.Label(top_row, text=f"{score:.1f}",
                     font=("Segoe UI", 20, "bold"),
                     bg=CARD_BG, fg=rank_col).pack(side="right", padx=(10, 0))

            # Score bar
            bar_bg = tk.Frame(inner, bg=ACCENT, height=6)
            bar_bg.pack(fill="x", pady=(8, 0))
            bar_bg.pack_propagate(False)
            tk.Frame(bar_bg, bg=rank_col, height=6).place(relwidth=pct, relheight=1.0)

            HOVER_BG    = "#253656"
            SELECTED_BG = "#1f4068"

            self._card_widgets.append({"card": card, "inner": inner, "top_row": top_row, "chip_row": chip_row, "config": config})

            for w in _all_children(card):
                w.bind("<Enter>",    lambda e, i=i: self._card_hover(i, True))
                w.bind("<Leave>",    lambda e, i=i: self._card_hover(i, False))
                w.bind("<Button-1>", lambda e, i=i: self._card_click(i))
                try: w.configure(cursor="hand2")
                except: pass

    def _set_card_colors(self, idx, bg):
        if idx >= len(self._card_widgets):
            return
        d = self._card_widgets[idx]
        d["card"].configure(bg=bg)
        d["inner"].configure(bg=bg)
        d["top_row"].configure(bg=bg)
        d["chip_row"].configure(bg=bg)

    def _card_hover(self, idx, enter):
        if idx == self._selected_index:
            return
        if enter:
            self._set_card_colors(idx, "#253656")
        else:
            self._set_card_colors(idx, CARD_BG)

    def _card_click(self, idx):
        if self._selected_index is not None and self._selected_index != idx:
            self._set_card_colors(self._selected_index, CARD_BG)
        self._selected_index = idx
        self._set_card_colors(idx, "#1f4068")
        self._show_reasoning(self._card_widgets[idx]["config"], self._get_weights())
        self._nb.select(self._tab_reason)

    # ── Tab: Evolution Chart ──────────────────────────────────────────────────

    def _build_chart_tab(self, parent):
        self._fig, self._ax = plt.subplots(figsize=(8, 4.8))
        self._ax2 = None   # secondary axis — created once, reused
        self._fig.patch.set_facecolor(BG)
        self._ax.set_facecolor(PANEL)
        self._ax.set_title("Run the GA to see the evolution chart", color=SUBTEXT, fontsize=11)
        for spine in self._ax.spines.values():
            spine.set_edgecolor(ACCENT)
        self._canvas = FigureCanvasTkAgg(self._fig, master=parent)
        self._canvas.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def _update_chart(self, history, mutation_history):
        # Clear primary axis
        self._ax.clear()
        self._ax.set_facecolor(PANEL)

        # Remove old secondary axis completely before creating a new one
        if self._ax2 is not None:
            self._ax2.remove()
        self._ax2 = self._ax.twinx()

        gens = range(1, len(history) + 1)

        self._ax.plot(gens, history, color=HIGHLIGHT, linewidth=2.5,
                      marker="o", markersize=4, label="Best Fitness", zorder=3)
        self._ax.fill_between(gens, history, min(history), alpha=0.15, color=HIGHLIGHT)
        self._ax.set_ylabel("Best Fitness", color=HIGHLIGHT, fontsize=10)
        self._ax.tick_params(axis="y", colors=HIGHLIGHT)
        self._ax.tick_params(axis="x", colors=SUBTEXT)
        self._ax.set_xlabel("Generation", color=SUBTEXT, fontsize=10)
        self._ax.grid(axis="y", color=ACCENT, alpha=0.4, linestyle="--")

        self._ax2.plot(gens, mutation_history, color=WARN, linewidth=1.5,
                       linestyle="--", alpha=0.85, label="Mutation Rate")
        self._ax2.set_ylabel("Mutation Rate", color=WARN, fontsize=10)
        self._ax2.tick_params(axis="y", colors=WARN)
        self._ax2.set_ylim(0, 0.75)
        self._ax2.set_facecolor(PANEL)

        self._ax.set_title("Fitness Evolution  +  Adaptive Mutation Rate",
                            color=TEXT, fontsize=12, pad=10)
        for spine in self._ax.spines.values():  spine.set_edgecolor(ACCENT)
        for spine in self._ax2.spines.values(): spine.set_edgecolor(ACCENT)

        lines1, labels1 = self._ax.get_legend_handles_labels()
        lines2, labels2 = self._ax2.get_legend_handles_labels()
        self._ax.legend(lines1 + lines2, labels1 + labels2,
                        loc="lower right", facecolor=PANEL,
                        labelcolor=TEXT, fontsize=9, framealpha=0.9)
        self._fig.tight_layout()
        self._canvas.draw()

    # ── Tab: Constraints ──────────────────────────────────────────────────────

    def _build_csp_tab(self, parent):
        tk.Label(parent, text="Constraint Validation",
                 font=FONT_H, bg=BG, fg=TEXT).pack(pady=(14, 2), padx=15, anchor="w")
        tk.Label(parent, text="Shows which hard constraints the selected config satisfies",
                 font=FONT_S, bg=BG, fg=SUBTEXT).pack(padx=15, anchor="w")
        self._csp_frame = tk.Frame(parent, bg=BG)
        self._csp_frame.pack(fill="both", expand=True, padx=15, pady=10)
        self._empty_state(self._csp_frame, "🔒", "No config selected",
                          "Click a result card to validate its constraints")

    def _render_csp(self, config):
        for w in self._csp_frame.winfo_children():
            w.destroy()

        # Config chips
        summary = tk.Frame(self._csp_frame, bg=BG)
        summary.pack(fill="x", pady=(0, 14))
        tk.Label(summary, text="Config:", font=FONT_S, bg=BG, fg=SUBTEXT).pack(side="left", padx=(0, 8))
        for var in VARIABLES:
            chip = tk.Frame(summary, bg=ACCENT, padx=6, pady=2)
            chip.pack(side="left", padx=3)
            tk.Label(chip, text=f"{COMPONENT_ICONS.get(var,'')} {config[var]}",
                     font=("Segoe UI", 9), bg=ACCENT, fg=TEXT).pack()

        rule_defs = [
            ("Electric engine",  "requires No transmission"),
            ("Turbo",            "only allowed with VTEC or V8 engine"),
            ("Servis tires",     "require Alloy or Steel rims"),
            ("V8 engine",        "requires Sport or Heavy-Duty suspension"),
            ("Turbo",            "requires Sport or Heavy-Duty suspension"),
            ("Carbon rims",      "require Sport or Heavy-Duty suspension"),
            ("Electric engine",  "cannot have Turbo"),
        ]

        passed = sum(1 for c in CONSTRAINTS if c(config))
        total  = len(CONSTRAINTS)

        # Summary bar
        bar_panel = tk.Frame(self._csp_frame, bg=PANEL, pady=10, padx=14)
        bar_panel.pack(fill="x", pady=(0, 12))
        tk.Label(bar_panel,
                 text=f"Constraints passed: {passed} / {total}",
                 font=("Segoe UI", 11, "bold"), bg=PANEL,
                 fg=SUCCESS if passed == total else WARN).pack(anchor="w")
        prog_bg = tk.Frame(bar_panel, bg=ACCENT, height=8)
        prog_bg.pack(fill="x", pady=(6, 0))
        prog_bg.pack_propagate(False)
        tk.Frame(prog_bg, bg=SUCCESS if passed == total else WARN, height=8
                 ).place(relwidth=passed/total, relheight=1.0)

        # Rule rows
        for i, ((subject, predicate), constraint) in enumerate(zip(rule_defs, CONSTRAINTS)):
            ok     = constraint(config)
            row_bg = PANEL
            row    = tk.Frame(self._csp_frame, bg=row_bg, pady=8, padx=14)
            row.pack(fill="x", pady=3)

            icon_lbl = "✅" if ok else "❌"
            s_color  = SUCCESS if ok else HIGHLIGHT
            status   = "PASS" if ok else "FAIL"

            left_col = tk.Frame(row, bg=row_bg)
            left_col.pack(side="left", fill="x", expand=True)
            title_row = tk.Frame(left_col, bg=row_bg)
            title_row.pack(anchor="w")
            tk.Label(title_row, text=icon_lbl, font=("Segoe UI", 13),
                     bg=row_bg).pack(side="left", padx=(0, 8))
            tk.Label(title_row, text=subject, font=("Segoe UI", 10, "bold"),
                     bg=row_bg, fg=TEXT).pack(side="left")
            tk.Label(title_row, text=f"  {predicate}", font=("Segoe UI", 10),
                     bg=row_bg, fg=SUBTEXT).pack(side="left")
            tk.Label(row, text=status, font=("Segoe UI", 9, "bold"),
                     bg=s_color, fg="white", padx=8, pady=2).pack(side="right", padx=4)

    # ── Tab: Reasoning ────────────────────────────────────────────────────────

    def _build_reason_tab(self, parent):
        top_bar = tk.Frame(parent, bg=BG)
        top_bar.pack(fill="x", padx=15, pady=(14, 6))
        tk.Label(top_bar, text="Configuration Reasoning",
                 font=FONT_H, bg=BG, fg=TEXT).pack(side="left")
        tk.Label(top_bar, text="Click any result card to load",
                 font=FONT_S, bg=BG, fg=SUBTEXT).pack(side="right")

        outer = tk.Frame(parent, bg=BG)
        outer.pack(fill="both", expand=True, padx=15, pady=(0, 12))

        reason_canvas = tk.Canvas(outer, bg=BG, highlightthickness=0)
        r_scroll = ttk.Scrollbar(outer, orient="vertical", command=reason_canvas.yview)
        reason_canvas.configure(yscrollcommand=r_scroll.set)
        r_scroll.pack(side="right", fill="y")
        reason_canvas.pack(side="left", fill="both", expand=True)

        self._reason_outer = tk.Frame(reason_canvas, bg=BG)
        r_win = reason_canvas.create_window((0, 0), window=self._reason_outer, anchor="nw")

        def _on_r_frame(e):
            reason_canvas.configure(scrollregion=reason_canvas.bbox("all"))
        def _on_r_canvas(e):
            reason_canvas.itemconfig(r_win, width=e.width)

        self._reason_outer.bind("<Configure>", _on_r_frame)
        reason_canvas.bind("<Configure>", _on_r_canvas)
        reason_canvas.bind_all("<MouseWheel>",
            lambda e: reason_canvas.yview_scroll(-1*(e.delta//120), "units"))

        self._empty_state(self._reason_outer, "💡", "No config selected",
                          "Click a result card to see reasoning here")

    def _show_reasoning(self, config, weights):
        for w in self._reason_outer.winfo_children():
            w.destroy()

        stats = _aggregate_stats(config)
        score = evaluate_fitness(config, weights)

        # Score banner
        banner = tk.Frame(self._reason_outer, bg=CARD_BG, pady=14)
        banner.pack(fill="x", pady=(0, 12))
        tk.Label(banner, text=f"🏆  Fitness Score: {score:.2f}",
                 font=("Segoe UI", 16, "bold"), bg=CARD_BG, fg=WARN).pack(side="left", padx=20)

        for stat_name, val, color in [
            ("Performance",  stats["performance"],  HIGHLIGHT),
            ("Stability",    stats["stability"],    SUCCESS),
            ("Efficiency",   stats["efficiency"],   BLUE),
            ("Aerodynamics", stats["aerodynamics"], WARN),
            ("Weight",       stats["weight"],       SUBTEXT),
        ]:
            pill = tk.Frame(banner, bg=ACCENT, padx=10, pady=4)
            pill.pack(side="left", padx=6)
            tk.Label(pill, text=str(val), font=("Segoe UI", 12, "bold"),
                     bg=ACCENT, fg=color).pack()
            tk.Label(pill, text=stat_name, font=("Segoe UI", 8),
                     bg=ACCENT, fg=SUBTEXT).pack()

        tk.Label(self._reason_outer, text="Component Breakdown",
                 font=FONT_H, bg=BG, fg=TEXT).pack(anchor="w", pady=(4, 8))

        highlights = {
            "engine":       {"V8":       ("🔥 Max performance engine", HIGHLIGHT),
                             "Electric": ("⚡ Top efficiency, zero emissions", SUCCESS),
                             "VTEC":     ("🏎️  Balanced power and efficiency", BLUE)},
            "tires":        {"Pirelli":  ("🏁 Maximum grip and stability", SUCCESS),
                             "Michelin": ("✅ Reliable all-round performer", BLUE),
                             "Servis":   ("🪙 Budget option, lower grip", SUBTEXT)},
            "rims":         {"Carbon":   ("🚀 Ultra-light, high stiffness", SUCCESS),
                             "Alloy":    ("⚖️  Good weight-to-strength ratio", BLUE),
                             "Steel":    ("🏋️  Heavy but very durable", SUBTEXT)},
            "suspension":   {"Sport":      ("🏋️  Handles stress and cornering well", SUCCESS),
                             "Heavy-Duty": ("💪 Built for heavy, powerful engines", BLUE),
                             "Standard":   ("🔧 Basic, suits lighter builds", SUBTEXT)},
            "transmission": {"Manual":    ("🎮 Driver-controlled, better efficiency", BLUE),
                             "Automatic": ("🤖 Smooth shifting, better acceleration", SUCCESS),
                             "None":      ("🔌 Not needed — Electric drivetrain", SUBTEXT)},
            "body_type":    {"Coupe": ("✈️  Best aerodynamics in the lineup", SUCCESS),
                             "Sedan": ("🚗 Good aero-weight balance", BLUE),
                             "SUV":   ("🏔️  Poor aero, compensates with size", SUBTEXT)},
            "turbo":        {"Yes": ("💨 Significant power boost", HIGHLIGHT),
                             "No":  ("🔇 Saves weight, improves efficiency", SUBTEXT)},
        }

        grid = tk.Frame(self._reason_outer, bg=BG)
        grid.pack(fill="x")

        for idx, var in enumerate(VARIABLES):
            val          = config[var]
            note, color  = highlights.get(var, {}).get(val, ("", SUBTEXT))
            icon         = COMPONENT_ICONS.get(var, "")

            card = tk.Frame(grid, bg=CARD_BG, padx=12, pady=10)
            card.grid(row=idx // 4, column=idx % 4, padx=5, pady=5, sticky="nsew")

            tk.Label(card, text=f"{icon}  {var.replace('_',' ').title()}",
                     font=("Segoe UI", 8), bg=CARD_BG, fg=SUBTEXT).pack(anchor="w")
            tk.Label(card, text=val, font=("Segoe UI", 12, "bold"),
                     bg=CARD_BG, fg=color).pack(anchor="w", pady=(2, 4))
            tk.Label(card, text=note, font=("Segoe UI", 8),
                     bg=CARD_BG, fg=SUBTEXT, wraplength=160, justify="left").pack(anchor="w")

        for col in range(4):
            grid.columnconfigure(col, weight=1)

        self._render_csp(config)

    # ── Close ─────────────────────────────────────────────────────────────────

    def _on_close(self):
        self._stop_flag = True
        self.destroy()
        sys.exit(0)

    # ── GA Runner ─────────────────────────────────────────────────────────────

    def _reset_results_to_empty(self):
        self._top_configs = []
        self._history     = []
        self._card_widgets = []
        self._selected_index = None

        for frame in (self._results_frame, self._reason_outer, self._csp_frame):
            for w in frame.winfo_children():
                w.destroy()

        self._empty_state(self._results_frame, "🧬", "Run the GA to see results",
                          "Configure your priorities and press Run GA")
        self._empty_state(self._reason_outer, "💡", "No config selected",
                          "Click a result card to see reasoning here")
        self._empty_state(self._csp_frame, "🔒", "No config selected",
                          "Click a result card to validate its constraints")

        for key in self._stat_vars:
            self._stat_vars[key].set("—")

    def _reset_sliders(self):
        for key, var in self._sliders.items():
            var.set(0.5)
            if key in self._slider_labels:
                self._slider_labels[key].config(text="0.50")
        self._gen_var.set(30)
        self._gen_disp.config(text="30 gens")
        self._reset_results_to_empty()

    def _get_weights(self):
        return {k: v.get() for k, v in self._sliders.items()}

    def _start_ga(self):
        if self._is_running:
            return
        self._is_running = True
        self._stop_flag  = False
        self._run_btn.configure(state="disabled", text="⏳  Running...")
        self._status_var.set("Generating valid configs via CSP...")
        self._progress["value"] = 0
        threading.Thread(target=self._run_pipeline, daemon=True).start()

    def _run_pipeline(self):
        weights     = self._get_weights()
        generations = self._gen_var.get()

        population = generate_valid_configs(80)
        if self._stop_flag: return
        self.after(0, lambda: self._progress.configure(value=20))
        self.after(0, lambda: self._status_var.set(f"Running GA — {generations} generations..."))

        def progress_cb(gen, fitness):
            if self._stop_flag: return
            pct = 20 + int((gen / generations) * 75)
            self.after(0, lambda p=pct, g=gen, f=fitness: (
                self._progress.configure(value=p),
                self._status_var.set(f"Generation {g}/{generations}  —  Best: {f:.1f}")
            ))

        final_pop, history, mut_hist = run_ga(
            population, weights, generations=generations,
            progress_callback=progress_cb,
            stop_flag=lambda: self._stop_flag
        )
        if self._stop_flag: return
        self.after(0, lambda: self._finish(final_pop, history, mut_hist, weights, len(population)))

    def _finish(self, final_pop, history, mut_hist, weights, n_configs):
        self._top_configs = pick_diverse_top(final_pop, weights, n=5, min_diff=2)
        self._history     = history

        self._render_results(self._top_configs, weights)
        self._update_chart(history, mut_hist)

        best    = self._top_configs[0]
        improve = history[-1] - history[0]
        self._update_stats(evaluate_fitness(best, weights), len(history), n_configs, improve)
        self._show_reasoning(best, weights)

        self._progress["value"] = 100
        self._status_var.set(f"✅ Done!  Best fitness: {evaluate_fitness(best, weights):.1f}")
        self._run_btn.configure(state="normal", text="▶   Run GA")
        self._is_running = False


# ── Helpers ────────────────────────────────────────────────────────────────────

def _all_children(widget):
    yield widget
    for child in widget.winfo_children():
        yield from _all_children(child)

def _recolor(widget, color):
    try: widget.configure(bg=color)
    except: pass


# ── Entry Point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = CarConfigApp()
    app.mainloop()