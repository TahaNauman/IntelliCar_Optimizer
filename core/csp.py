# ─────────────────────────────────────────────
#  core/csp.py  –  OR-Tools CSP solver + manual validator
# ─────────────────────────────────────────────

from ortools.sat.python import cp_model
from core.data import DOMAINS, CONSTRAINTS, VARIABLES
import random


# ── Index maps: component name → integer (OR-Tools works with integers) ────────
DOMAIN_INDEX = {var: {val: i for i, val in enumerate(DOMAINS[var])} for var in VARIABLES}
INDEX_DOMAIN = {var: {i: val for i, val in enumerate(DOMAINS[var])} for var in VARIABLES}


def is_valid(config: dict) -> bool:
    """
    Fast manual constraint checker.
    Used after GA mutation/crossover to repair or discard invalid configs.
    Returns True if ALL hard constraints are satisfied.
    """
    return all(constraint(config) for constraint in CONSTRAINTS)


def generate_valid_configs(n: int = 100) -> list[dict]:
    """
    Use OR-Tools CP-SAT to enumerate valid car configurations.
    Returns up to `n` valid configurations as a list of dicts.
    """
    model = cp_model.CpModel()

    # Create OR-Tools integer variables — one per component
    vars_ = {
        var: model.NewIntVar(0, len(DOMAINS[var]) - 1, var)
        for var in VARIABLES
    }

    # ── Encode hard constraints into OR-Tools ──────────────────────────────────

    # Helper to get index of a value for a variable
    def idx(var, val):
        return DOMAIN_INDEX[var][val]

    # 1. Electric → transmission == None
    elec = idx("engine", "Electric")
    none_trans = idx("transmission", "None")
    # If engine == Electric → transmission must be None
    b_elec = model.NewBoolVar("is_electric")
    model.Add(vars_["engine"] == elec).OnlyEnforceIf(b_elec)
    model.Add(vars_["engine"] != elec).OnlyEnforceIf(b_elec.Not())
    model.Add(vars_["transmission"] == none_trans).OnlyEnforceIf(b_elec)
    model.Add(vars_["transmission"] != none_trans).OnlyEnforceIf(b_elec.Not())

    # 2. Electric → turbo must be No
    no_turbo = idx("turbo", "No")
    model.Add(vars_["turbo"] == no_turbo).OnlyEnforceIf(b_elec)

    # 3. Servis tires → rims in [Alloy, Steel]
    servis = idx("tires", "Servis")
    alloy = idx("rims", "Alloy")
    steel = idx("rims", "Steel")
    b_servis = model.NewBoolVar("is_servis")
    model.Add(vars_["tires"] == servis).OnlyEnforceIf(b_servis)
    model.Add(vars_["tires"] != servis).OnlyEnforceIf(b_servis.Not())
    # When Servis: rims must be Alloy or Steel (not Carbon)
    # We encode: rims != Carbon when Servis
    carbon = idx("rims", "Carbon")
    model.Add(vars_["rims"] != carbon).OnlyEnforceIf(b_servis)

    # 4. V8 → suspension in [Heavy-Duty, Sport]
    v8 = idx("engine", "V8")
    sport_susp = idx("suspension", "Sport")
    heavy_susp = idx("suspension", "Heavy-Duty")
    b_v8 = model.NewBoolVar("is_v8")
    model.Add(vars_["engine"] == v8).OnlyEnforceIf(b_v8)
    model.Add(vars_["engine"] != v8).OnlyEnforceIf(b_v8.Not())
    b_v8_susp_ok = model.NewBoolVar("v8_susp_ok")
    model.AddLinearConstraint(vars_["suspension"], sport_susp, heavy_susp).OnlyEnforceIf(b_v8)

    # 5. Turbo Yes → suspension in [Sport, Heavy-Duty]
    yes_turbo = idx("turbo", "Yes")
    b_turbo = model.NewBoolVar("has_turbo")
    model.Add(vars_["turbo"] == yes_turbo).OnlyEnforceIf(b_turbo)
    model.Add(vars_["turbo"] != yes_turbo).OnlyEnforceIf(b_turbo.Not())
    model.AddLinearConstraint(vars_["suspension"], sport_susp, heavy_susp).OnlyEnforceIf(b_turbo)

    # 6. Carbon rims → suspension in [Sport, Heavy-Duty]
    b_carbon = model.NewBoolVar("has_carbon")
    model.Add(vars_["rims"] == carbon).OnlyEnforceIf(b_carbon)
    model.Add(vars_["rims"] != carbon).OnlyEnforceIf(b_carbon.Not())
    model.AddLinearConstraint(vars_["suspension"], sport_susp, heavy_susp).OnlyEnforceIf(b_carbon)

    # ── Enumerate solutions ────────────────────────────────────────────────────
    solver = cp_model.CpSolver()
    solver.parameters.enumerate_all_solutions = True

    configs = []

    class SolutionCollector(cp_model.CpSolverSolutionCallback):
        def __init__(self, variables, limit):
            cp_model.CpSolverSolutionCallback.__init__(self)
            self._vars = variables
            self._limit = limit
            self._count = 0

        def on_solution_callback(self):
            if self._count >= self._limit:
                self.StopSearch()
                return
            config = {
                var: INDEX_DOMAIN[var][self.Value(self._vars[var])]
                for var in VARIABLES
            }
            configs.append(config)
            self._count += 1

    collector = SolutionCollector(vars_, n)
    solver.Solve(model, collector)

    # Shuffle so we don't always start with the same population
    random.shuffle(configs)
    return configs


if __name__ == "__main__":
    configs = generate_valid_configs(10)
    print(f"Generated {len(configs)} valid configurations:\n")
    for i, c in enumerate(configs, 1):
        print(f"  Config {i}: {c}")
        print(f"  Valid check: {is_valid(c)}\n")