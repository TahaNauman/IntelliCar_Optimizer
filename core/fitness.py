# ─────────────────────────────────────────────
#  core/fitness.py  –  Fitness evaluation
# ─────────────────────────────────────────────

from core.data import COMPONENT_STATS


def evaluate_fitness(config: dict, weights: dict) -> float:
    """
    Compute the fitness score of a configuration.

    weights: dict with keys 'performance', 'stability', 'efficiency', 'aerodynamics'
             Each value is a float 0.0–1.0 (user preference sliders)
             They are normalized internally so they don't need to sum to 1.

    Formula:
        fitness = w_perf  * performance
                + w_stab  * stability
                + w_eff   * efficiency
                + w_aero  * aerodynamics
                - penalty (imbalance between high power and low stability)
    """
    stats = _aggregate_stats(config)

    perf  = stats["performance"]
    stab  = stats["stability"]
    eff   = stats["efficiency"]
    aero  = stats["aerodynamics"]
    total_weight = stats["weight"]

    # Normalize user weights so they sum to 1
    w = weights.copy()
    total = sum(w.values()) or 1
    for k in w:
        w[k] /= total

    raw_score = (
        w.get("performance", 0.25) * perf +
        w.get("stability",   0.25) * stab +
        w.get("efficiency",  0.25) * eff  +
        w.get("aerodynamics",0.25) * aero
    )

    # Penalty: heavy car with low stability is dangerous
    imbalance_penalty = 0.0
    if total_weight > 150 and stab < 80:
        imbalance_penalty = 8.0

    # Penalty: high performance but terrible efficiency
    if perf > 120 and eff < 30:
        imbalance_penalty += 5.0

    fitness = raw_score - imbalance_penalty
    return round(max(fitness, 0.0), 4)


def _aggregate_stats(config: dict) -> dict:
    """Sum up all stat contributions from every selected component."""
    totals = {"performance": 0, "stability": 0, "efficiency": 0,
              "aerodynamics": 0, "weight": 0}

    for component, chosen in config.items():
        stats = COMPONENT_STATS.get(component, {}).get(chosen, {})
        for stat, value in stats.items():
            if stat in totals:
                totals[stat] += value

    return totals


def explain_fitness(config: dict, weights: dict) -> str:
    """
    Generate a human-readable explanation of why a config scored well.
    Used in the GUI 'reasoning' panel.
    """
    stats = _aggregate_stats(config)
    score = evaluate_fitness(config, weights)

    lines = [
        f"🏆 Fitness Score: {score:.2f}",
        f"",
        f"📊 Aggregated Stats:",
        f"   Performance  : {stats['performance']}",
        f"   Stability    : {stats['stability']}",
        f"   Efficiency   : {stats['efficiency']}",
        f"   Aerodynamics : {stats['aerodynamics']}",
        f"   Total Weight : {stats['weight']}",
        f"",
        f"🔍 Component Choices:",
    ]

    highlights = {
        "engine":       {"V8": "🔥 V8 gives max performance",
                         "Electric": "⚡ Electric = top efficiency",
                         "VTEC":     "🏎️  VTEC balances power & efficiency"},
        "tires":        {"Pirelli": "🏁 Pirelli delivers max grip",
                         "Michelin": "✅ Michelin is a solid all-rounder",
                         "Servis":   "🪙 Servis is budget-friendly"},
        "rims":         {"Carbon": "🚀 Carbon rims are ultra-light",
                         "Alloy":  "⚖️  Alloy rims balance weight & strength",
                         "Steel":  "🏋️  Steel rims are heavy but durable"},
        "suspension":   {"Sport": "🏋️  Sport suspension handles stress well",
                         "Heavy-Duty": "💪 Heavy-Duty handles big engines",
                         "Standard": "🔧 Standard suspension for everyday use"},
        "transmission": {"Manual": "🎮 Manual gives driver control",
                         "Automatic": "🤖 Automatic for smooth driving",
                         "None": "🔌 No transmission needed (Electric)"},
        "body_type":    {"Coupe": "✈️  Coupe has the best aerodynamics",
                         "Sedan": "🚗 Sedan is a good aero-weight balance",
                         "SUV":   "🏔️  SUV trades aero for cargo space"},
        "turbo":        {"Yes": "💨 Turbo boosts performance significantly",
                         "No":  "🔇 No turbo keeps weight and cost down"},
    }

    for comp, val in config.items():
        note = highlights.get(comp, {}).get(val, "")
        lines.append(f"   {comp.capitalize():<14}: {val}  {note}")

    return "\n".join(lines)