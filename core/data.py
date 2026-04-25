# ─────────────────────────────────────────────
#  core/data.py  –  All domains & constraint rules
# ─────────────────────────────────────────────

# ── Component Domains ──────────────────────────────────────────────────────────
DOMAINS = {
    "engine":       ["VTEC", "V8", "Electric"],
    "tires":        ["Servis", "Michelin", "Pirelli"],
    "rims":         ["Alloy", "Steel", "Carbon"],
    "suspension":   ["Standard", "Sport", "Heavy-Duty"],
    "transmission": ["Manual", "Automatic", "None"],
    "body_type":    ["Sedan", "SUV", "Coupe"],
    "turbo":        ["Yes", "No"],
}

# ── Performance Attributes ─────────────────────────────────────────────────────
# Each component contributes to: performance, stability, efficiency, aerodynamics
# Scale: 0–10

COMPONENT_STATS = {
    # Engine: big spread between options so choice matters a lot
    "engine": {
        "VTEC":     {"performance": 55, "weight": 30, "efficiency": 60},
        "V8":       {"performance": 90, "weight": 75, "efficiency": 20},
        "Electric": {"performance": 45, "weight": 25, "efficiency": 95},
    },
    # Tires: wide gap so Pirelli vs Servis is a meaningful trade-off
    "tires": {
        "Servis":   {"stability": 25, "weight": 20},
        "Michelin": {"stability": 60, "weight": 30},
        "Pirelli":  {"stability": 90, "weight": 40},
    },
    # Rims: Carbon is clearly best for performance but penalised by weight rules
    "rims": {
        "Alloy":  {"weight": 30, "stability": 45},
        "Steel":  {"weight": 60, "stability": 20},
        "Carbon": {"weight": 10, "stability": 80},
    },
    # Suspension: each option has a clear niche — Standard vs Sport vs Heavy-Duty
    "suspension": {
        "Standard":   {"stability": 20, "weight": 25},
        "Sport":      {"stability": 70, "weight": 35},
        "Heavy-Duty": {"stability": 55, "weight": 70},
    },
    # Transmission: Manual rewards efficiency-focused builds, Automatic rewards performance
    "transmission": {
        "Manual":    {"efficiency": 65, "performance": 40},
        "Automatic": {"efficiency": 35, "performance": 60},
        "None":      {"efficiency": 0,  "performance": 0},   # Electric only
    },
    # Body: Coupe is aero king, SUV is the clear loser aerodynamically
    "body_type": {
        "Sedan": {"aerodynamics": 55, "weight": 35},
        "SUV":   {"aerodynamics": 15, "weight": 65},
        "Coupe": {"aerodynamics": 85, "weight": 25},
    },
    # Turbo: meaningful performance boost but real efficiency cost
    "turbo": {
        "Yes": {"performance": 30, "efficiency": -25, "weight": 10},
        "No":  {"performance": 0,  "efficiency": 0,   "weight": 0},
    },
}

# ── Hard Constraints (as plain rules — used by manual validator) ───────────────
# Each constraint is a function: config (dict) -> bool (True = constraint satisfied)

CONSTRAINTS = [
    # 1. Electric engine → no transmission
    lambda c: c["transmission"] == "None" if c["engine"] == "Electric" else c["transmission"] != "None",

    # 2. Turbo only with VTEC or V8 (not Electric)
    lambda c: c["turbo"] == "No" if c["engine"] == "Electric" else True,

    # 3. Servis tires → must use Alloy or Steel rims (not Carbon — too high-spec)
    lambda c: c["rims"] in ["Alloy", "Steel"] if c["tires"] == "Servis" else True,

    # 4. V8 engine → must have Heavy-Duty or Sport suspension (heavy engine)
    lambda c: c["suspension"] in ["Heavy-Duty", "Sport"] if c["engine"] == "V8" else True,

    # 5. Turbo Yes → must have Sport or Heavy-Duty suspension (handles stress)
    lambda c: c["suspension"] in ["Sport", "Heavy-Duty"] if c["turbo"] == "Yes" else True,

    # 6. Carbon rims → only with Sport or Heavy-Duty suspension (high performance pairing)
    lambda c: c["suspension"] in ["Sport", "Heavy-Duty"] if c["rims"] == "Carbon" else True,

    # 7. Electric engine → no turbo (redundant but explicit)
    lambda c: c["turbo"] == "No" if c["engine"] == "Electric" else True,
]

VARIABLES = list(DOMAINS.keys())