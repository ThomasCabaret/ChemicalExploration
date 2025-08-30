"""
Reaction-network solver (math-first, chemistry-agnostic), with a clean ChemProvider API.

- Works end-to-end with synthetic random data (no real chemistry required).
- All chemistry-like numbers are fetched via ChemProvider getters (easy to swap later).
- Incremental scope expansion with clear logs per round.
- MILP (via python-mip/CBC) selects a subnetwork that is:
    * F-generated and closed,
    * autocatalytic with respect to species X,
    * kinetically dominant vs off-network reactions for actually-consumed species,
    * optionally producing target species (e.g., "amphiphiles") by demand and/or objective.
- Best network is chosen by a continuous dominance-based score plus gain on X and targets.

Requirements:
  python -m pip install --upgrade pip
  python -m pip install numpy mip

Run:
  python this_file.py
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Set, Dict, Optional, Tuple
import time
import numpy as np

try:
    import pulp
    from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, LpContinuous, lpSum, PULP_CBC_CMD, LpStatus
except Exception as e:
    raise RuntimeError("Please install PuLP: pip install pulp") from e


# ------------------------------ ChemProvider API ------------------------------

class ChemProvider:
    """
    Abstract provider for all chemistry-facing data. Replace SyntheticChemProvider
    with a real implementation later without touching the solver.

    Species are indexed 0..m-1; reactions 0..n-1.
    """

    # --- Topology / sizes ---
    def species_count(self) -> int:
        raise NotImplementedError

    def reaction_count(self) -> int:
        raise NotImplementedError

    # --- Core sets ---
    def food_set(self) -> Set[int]:
        raise NotImplementedError

    def autocatalyst_index(self) -> int:
        raise NotImplementedError

    def target_species(self) -> Set[int]:
        """Species we want net production for (e.g., amphiphiles)."""
        return set()

    # --- Global parameters and budgets ---
    def alpha_dominance(self) -> float:
        """Dominance factor alpha (>1)."""
        return 5.0

    def max_selected_reactions(self) -> Optional[int]:
        """Optional hard cap on number of selected reactions."""
        return None

    def food_budget(self) -> np.ndarray:
        """Length-m vector of allowed net consumption for food species (>=0), else 0."""
        raise NotImplementedError

    # --- Column accessors for a given scope (list of reaction indices) ---
    def stoichiometry_submatrix(self, scope: List[int]) -> np.ndarray:
        """Returns S[:, scope] of shape (m, k)."""
        raise NotImplementedError

    def capacity_upper_bounds(self, scope: List[int]) -> np.ndarray:
        """Returns U[scope] of length k."""
        raise NotImplementedError

    def thermo_allowed_mask(self, scope: List[int]) -> np.ndarray:
        """Boolean mask length k: True if reaction is thermodynamically allowed (e.g., delta_rG <= 0)."""
        raise NotImplementedError

    def consumes_submatrix(self, scope: List[int]) -> np.ndarray:
        """Returns consumes[:, scope] where consumes[i,j] = max(-S[i,j], 0)."""
        raise NotImplementedError

    def participants_lists(self, scope: List[int]) -> List[List[int]]:
        """For each reaction in scope, list of species indices with S[i,j] != 0."""
        raise NotImplementedError

    def uses_X_mask(self, scope: List[int]) -> np.ndarray:
        """Boolean mask length k: True if reaction consumes species X."""
        raise NotImplementedError

    # --- Helpers for scope management and priorities ---
    def global_allowed_mask(self) -> np.ndarray:
        """Boolean mask length n over all reactions (for initial prioritization)."""
        raise NotImplementedError

    def reaction_participants_all(self, j: int) -> List[int]:
        """Participants (species indices) for reaction j (global)."""
        raise NotImplementedError

    def delta_rG_all(self) -> np.ndarray:
        """Return delta_rG for all reactions (only used for heuristics/ordering)."""
        raise NotImplementedError


# ------------------------------ Synthetic provider ------------------------------

class SyntheticChemProvider(ChemProvider):
    """
    Synthetic random network with CHNOPS-like flavor (no chemistry needed).
    - Ensures at least one reaction consumes X and one produces X.
    - Marks a subset of species as "targets" (e.g., amphiphiles).
    - Sets a thermodynamic gate (delta_rG <= 0 => allowed).
    """

    def __init__(
        self,
        m: int = 60,
        n: int = 600,
        food_size: int = 6,
        target_size: int = 6,
        max_reactants: int = 2,
        max_products: int = 2,
        allow_nonspontaneous: bool = False,
        alpha: float = 5.0,
        max_selected: Optional[int] = 60,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self._m = m
        self._n = n
        self._alpha = float(alpha)
        self._max_selected = max_selected
        self._allow_nonspontaneous = bool(allow_nonspontaneous)

        # Food set and target species (disjoint as a default guideline)
        self._food = set(self.rng.choice(m, size=food_size, replace=False).tolist())
        non_food = [i for i in range(m) if i not in self._food]
        self._targets = set(self.rng.choice(non_food, size=target_size, replace=False).tolist()) if len(non_food) >= target_size else set()

        # Autocatalyst X (prefer non-food)
        self._X = int(self.rng.choice(non_food)) if non_food else int(self.rng.integers(0, m))

        # Stoichiometry S
        S = np.zeros((m, n), dtype=float)
        for j in range(n):
            r_count = self.rng.integers(1, max_reactants + 1)
            p_count = self.rng.integers(1, max_products + 1)
            reactants = self.rng.choice(m, size=r_count, replace=False)
            products = self.rng.choice(m, size=p_count, replace=False)
            for i in reactants:
                S[i, j] -= int(self.rng.integers(1, 3))
            for i in products:
                S[i, j] += int(self.rng.integers(1, 3))
        # Force some X consumers/producers
        consumers = self.rng.choice(n, size=max(1, n // 12), replace=False)
        producers = self.rng.choice(n, size=max(1, n // 12), replace=False)
        for j in consumers:
            S[self._X, j] -= 1.0
        for j in producers:
            S[self._X, j] += 1.0

        self._S = S
        self._U = self.rng.uniform(1.0, 10.0, size=n)
        self._delta_rG = self.rng.normal(-5.0, 10.0, size=n)
        self._allowed_global = np.ones(n, dtype=bool) if self._allow_nonspontaneous else (self._delta_rG <= 0.0)
        self._consumes = np.maximum(-self._S, 0.0)
        self._participants_all = [list(np.nonzero(self._S[:, j])[0]) for j in range(n)]
        self._usesX_all = (self._S[self._X, :] < 0.0)

        # Food budget
        self._food_budget = np.zeros(m, dtype=float)
        for f in self._food:
            self._food_budget[f] = float(self.rng.uniform(20.0, 80.0))

    # --- API impl ---
    def species_count(self) -> int:
        return self._m

    def reaction_count(self) -> int:
        return self._n

    def food_set(self) -> Set[int]:
        return set(self._food)

    def autocatalyst_index(self) -> int:
        return self._X

    def target_species(self) -> Set[int]:
        return set(self._targets)

    def alpha_dominance(self) -> float:
        return self._alpha

    def max_selected_reactions(self) -> Optional[int]:
        return self._max_selected

    def food_budget(self) -> np.ndarray:
        return self._food_budget.copy()

    def stoichiometry_submatrix(self, scope: List[int]) -> np.ndarray:
        return self._S[:, scope]

    def capacity_upper_bounds(self, scope: List[int]) -> np.ndarray:
        return self._U[scope]

    def thermo_allowed_mask(self, scope: List[int]) -> np.ndarray:
        return self._allowed_global[scope]

    def consumes_submatrix(self, scope: List[int]) -> np.ndarray:
        return self._consumes[:, scope]

    def participants_lists(self, scope: List[int]) -> List[List[int]]:
        return [self._participants_all[j] for j in scope]

    def uses_X_mask(self, scope: List[int]) -> np.ndarray:
        return self._usesX_all[scope]

    def global_allowed_mask(self) -> np.ndarray:
        return self._allowed_global

    def reaction_participants_all(self, j: int) -> List[int]:
        return self._participants_all[j]

    def delta_rG_all(self) -> np.ndarray:
        return self._delta_rG


# ------------------------------ Problem spec (objectives/constraints) ------------------------------

@dataclass
class ProblemSpec:
    """Holds problem-specific knobs while keeping the solver generic."""
    eps_gain_X: float = 1.0
    eps_active: float = 1e-3
    lambda_size: float = 1e-2
    target_min_demand: float = 0.0   # per-target minimal net production (can be 0)
    target_weight: float = 1.0       # weight in objective for net production of targets
    enable_solver_log: bool = False
    max_rounds: int = 6
    init_scope: int = 120
    scope_step: int = 80


# ------------------------------ Scope manager ------------------------------

class ScopeManager:
    """
    Maintains an incremental reaction scope.
    Heuristic: prioritize allowed reactions that touch Food, X, or targets; then expand
    around species that proved active/consumed in the last solution.
    """

    def __init__(self, provider: ChemProvider, spec: ProblemSpec, seed: int = 0):
        self.p = provider
        self.spec = spec
        self.rng = np.random.default_rng(seed)
        self.n_total = self.p.reaction_count()
        self.in_scope: Set[int] = set()

        # Priority order over all reactions
        allowed = self.p.global_allowed_mask()
        touch_mask = np.zeros(self.n_total, dtype=bool)
        F = self.p.food_set()
        X = self.p.autocatalyst_index()
        T = self.p.target_species()
        for j in range(self.n_total):
            sp = self.p.reaction_participants_all(j)
            if (X in sp) or any(i in F for i in sp) or any(i in T for i in sp):
                touch_mask[j] = True

        base_rank = -self.p.delta_rG_all()  # more negative is better
        base_rank[~allowed] -= 1e6          # gate down disallowed
        base_rank = base_rank + 1e3 * touch_mask.astype(float)
        self.order = list(np.argsort(base_rank)[::-1])  # descending

    def initial_scope(self) -> List[int]:
        size = min(self.spec.init_scope, self.n_total)
        chosen = self.order[:size]
        self.in_scope = set(chosen)
        return list(self.in_scope)

    def expand(self, hint_species: Optional[Set[int]] = None) -> List[int]:
        remaining = [j for j in self.order if j not in self.in_scope]
        if not remaining:
            return list(self.in_scope)
        step = self.spec.scope_step

        if hint_species:
            touching = [j for j in remaining if any(i in hint_species for i in self.p.reaction_participants_all(j))]
            fillers = [j for j in remaining if j not in touching]
            pick = touching[:step]
            if len(pick) < step:
                pick += fillers[: (step - len(pick))]
        else:
            pick = remaining[:step]

        for j in pick:
            self.in_scope.add(j)
        return list(self.in_scope)

    def is_full(self) -> bool:
        return len(self.in_scope) >= self.n_total


# ------------------------------ MILP model (python-mip) ------------------------------

class SubnetworkMILP:
    """
    MILP for a given scope of reactions (PuLP + CBC).
    Variables:
      v_j >= 0     : reaction flux
      y_j in {0,1} : reaction selected
      x_i in {0,1} : species active
      a_i in {0,1} : species consumed in-network (triggers dominance)
    """

    def __init__(self, provider: ChemProvider, spec: ProblemSpec, scope: List[int]):
        self.p = provider
        self.spec = spec
        self.scope = list(scope)

    def solve(self) -> Dict[str, object]:
        p, spec, scope = self.p, self.spec, self.scope

        m = p.species_count()
        k = len(scope)
        S = p.stoichiometry_submatrix(scope)
        U = p.capacity_upper_bounds(scope)
        allowed = p.thermo_allowed_mask(scope)
        consumes = p.consumes_submatrix(scope)
        participants = p.participants_lists(scope)
        usesX = p.uses_X_mask(scope)

        F = p.food_set()
        X = p.autocatalyst_index()
        T = p.target_species()
        alpha = p.alpha_dominance()
        food_budget = p.food_budget()
        max_selected = p.max_selected_reactions()

        model = LpProblem("subnetwork", LpMaximize)

        v = [LpVariable(f"v_{j}", lowBound=0.0, upBound=float(U[j]), cat=LpContinuous) for j in range(k)]
        y = [LpVariable(f"y_{j}", lowBound=0, upBound=1, cat=LpBinary) for j in range(k)]
        x = [LpVariable(f"x_{i}", lowBound=0, upBound=1, cat=LpBinary) for i in range(m)]
        a = [LpVariable(f"a_{i}", lowBound=0, upBound=1, cat=LpBinary) for i in range(m)]

        for j in range(k):
            model += v[j] <= U[j] * y[j]
        for j in range(k):
            if not bool(allowed[j]):
                model += y[j] == 0
        for j in range(k):
            for i in participants[j]:
                model += y[j] <= x[i]

        for i in range(m):
            lhs = lpSum(S[i, jj] * v[jj] for jj in range(k))
            if i in F:
                model += lhs >= -float(food_budget[i])
            else:
                model += lhs >= 0.0

        gainX_expr = lpSum(S[X, j] * v[j] for j in range(k))
        model += gainX_expr >= self.spec.eps_gain_X
        idx_uses_x = [j for j in range(k) if usesX[j]]
        if idx_uses_x:
            model += lpSum(y[j] for j in idx_uses_x) >= 1
        else:
            return {"status": "NO_X_CONSUMER_IN_SCOPE", "solved": False}

        for i in range(m):
            Ci = consumes[i, :]
            if float(Ci.sum()) <= 0.0:
                continue
            c_in = lpSum(float(Ci[j]) * v[j] for j in range(k))
            c_off = lpSum(float(Ci[j]) * float(U[j]) * (1 - y[j]) for j in range(k))
            Mi = alpha * float(np.dot(Ci, U))
            model += c_in >= self.spec.eps_active * a[i]
            model += c_in >= alpha * c_off - Mi * (1 - a[i])

        target_terms = []
        for t in T:
            svt = lpSum(S[t, j] * v[j] for j in range(k))
            target_terms.append(svt)
            if self.spec.target_min_demand > 0.0:
                model += svt >= float(self.spec.target_min_demand)

        if max_selected is not None:
            model += lpSum(y) <= int(max_selected)

        obj = gainX_expr
        if target_terms:
            obj += self.spec.target_weight * lpSum(target_terms)
        obj -= self.spec.lambda_size * lpSum(y[j] for j in range(k))
        model += obj

        t0 = time.time()
        status_code = model.solve(PULP_CBC_CMD(msg=1 if self.spec.enable_solver_log else 0))
        t1 = time.time()
        status_str = LpStatus.get(status_code, str(status_code))

        # Only treat OPTIMAL as solved to avoid None values
        if status_str != "Optimal":
            return {
                "status": status_str, "solved": False, "time_s": t1 - t0,
                "gainX": None, "selected_reactions": [], "active_species": [],
                "consumed_species": [], "v": None, "y": None, "x": None, "a": None,
            }

        # Safe extractors (coerce None -> 0)
        def bval(var): 
            v = var.value()
            return 1 if (v is not None and v > 0.5) else 0
        def fval(var):
            v = var.value()
            return float(v) if v is not None else 0.0

        y_val = np.array([bval(var) for var in y], dtype=int)
        x_val = np.array([bval(var) for var in x], dtype=int)
        a_val = np.array([bval(var) for var in a], dtype=int)
        v_val = np.array([fval(var) for var in v], dtype=float)

        return {
            "status": status_str,
            "solved": True,
            "time_s": t1 - t0,
            "gainX": float(np.dot(S[self.p.autocatalyst_index(), :], v_val)),
            "selected_reactions": list(np.nonzero(y_val)[0]),
            "active_species": list(np.nonzero(x_val)[0]),
            "consumed_species": list(np.nonzero(a_val)[0]),
            "v": v_val, "y": y_val, "x": x_val, "a": a_val,
        }

# ------------------------------ Metrics and scoring ------------------------------

def dominance_metrics(provider: ChemProvider, scope: List[int], sol: Dict[str, object]) -> Dict[str, float]:
    """Compute continuous dominance ratios for consumed species."""
    if not sol.get("solved", False):
        return {"dominance_min_ratio": 0.0, "dominance_avg_ratio": 0.0}
    idx = list(scope)
    consumes = provider.consumes_submatrix(idx)
    U = provider.capacity_upper_bounds(idx)
    v = sol["v"]
    y = sol["y"]

    eps = 1e-9
    ratios = []
    for i in sol["consumed_species"]:
        Ci = consumes[i, :]
        if Ci.sum() <= 0.0:
            continue
        c_in = float(np.dot(Ci, v))
        c_off_max = float(np.dot(Ci * U, (1 - y)))
        ratios.append(c_in / (c_off_max + eps))
    if not ratios:
        return {"dominance_min_ratio": 0.0, "dominance_avg_ratio": 0.0}
    return {
        "dominance_min_ratio": float(np.min(ratios)),
        "dominance_avg_ratio": float(np.mean(ratios)),
    }


def network_score(sol: Dict[str, object],
                  dom: Dict[str, float],
                  lambda_size: float = 0.02,
                  gamma_min: float = 1.0,
                  gamma_avg: float = 0.3) -> float:
    """Continuous score combining gainX, dominance, and parsimony."""
    if not sol.get("solved", False):
        return -1e9
    k = int(np.sum(sol["y"])) if sol.get("y", None) is not None else 0
    return float(sol["gainX"]) + gamma_min * dom["dominance_min_ratio"] + gamma_avg * dom["dominance_avg_ratio"] - lambda_size * k


def debug_scope_snapshot(provider, scope, max_list=6):
    import numpy as np
    m = provider.species_count()
    k = len(scope)
    S = provider.stoichiometry_submatrix(scope)
    U = provider.capacity_upper_bounds(scope)
    allowed = provider.thermo_allowed_mask(scope)
    parts = provider.participants_lists(scope)
    usesX = provider.uses_X_mask(scope)
    X = provider.autocatalyst_index()
    T = provider.target_species()

    print(f"  [DBG] scope k={k} | allowed={int(allowed.sum())} | usesX={int(usesX.sum())}")
    if k == 0:
        return
    # capacity-style hints
    prod_cap_X = float(np.dot(np.maximum(S[X, :], 0.0), U))
    cons_cap_X = float(np.dot(np.maximum(-S[X, :], 0.0), U))
    print(f"  [DBG] X prod_cap={prod_cap_X:.3f}  cons_cap={cons_cap_X:.3f}")
    for t in sorted(T):
        prod_cap_t = float(np.dot(np.maximum(S[t, :], 0.0), U))
        print(f"  [DBG] target {t} prod_cap={prod_cap_t:.3f}")

    # list a few reactions (by largest U)
    idxs = list(np.argsort(-U))[:max_list]
    dRG_all = provider.delta_rG_all()
    for j in idxs:
        gidx = scope[j]
        dRG = (dRG_all[gidx] if dRG_all.size else float('nan'))
        sx = S[X, j] if m else 0.0
        print(f"    r{j}: allowed={bool(allowed[j])} usesX={bool(usesX[j])} U={U[j]:.2f} dGr'={dRG:.2f} parts={parts[j]} Sx={sx:+.0f}")


def debug_infeasibility_hints(provider, scope, spec):
    import numpy as np
    S = provider.stoichiometry_submatrix(scope)
    if S.shape[1] == 0:
        print("  [HINT] No reactions in scope.")
        return
    U = provider.capacity_upper_bounds(scope)
    allowed = provider.thermo_allowed_mask(scope)
    X = provider.autocatalyst_index()

    prod_cap_X_allowed = float(np.dot(np.maximum(S[X, :], 0.0), U * allowed.astype(float)))
    cons_cap_X_allowed = float(np.dot(np.maximum(-S[X, :], 0.0), U * allowed.astype(float)))
    print(f"  [DBG] (allowed-only) X prod_cap={prod_cap_X_allowed:.3f} cons_cap={cons_cap_X_allowed:.3f}")

    if spec.eps_gain_X > 0.0 and prod_cap_X_allowed <= 1e-9:
        print("  [HINT] No producer of X in allowed reactions, but eps_gain_X > 0 -> unsatisfiable.")

    usesX = provider.uses_X_mask(scope)
    if int(usesX.sum()) == 0:
        print("  [HINT] No reaction consumes X in scope -> autocatalysis trigger cannot be met.")

# ------------------------------ Driver (incremental scope with logs) ------------------------------

def search_best_network(provider: ChemProvider, spec: ProblemSpec, random_seed: int = 42) -> None:
    scope_mgr = ScopeManager(provider, spec, seed=random_seed)
    scope = scope_mgr.initial_scope()

    best = None  # (score, round_id, scope_size, sol, dom, scope_list)

    m, n = provider.species_count(), provider.reaction_count()
    F = provider.food_set()
    X = provider.autocatalyst_index()
    T = provider.target_species()

    print("== Reaction-network search (incremental scope) ==")
    print(f"Species m={m}, Reactions n={n}, Food={sorted(list(F))}, X={X}, Targets={sorted(list(T)) or 'None'}")
    print("Dominance alpha=", provider.alpha_dominance(), " | thermo gating handled by provider.")
    print("--------------------------------------------------")

    hint_species: Optional[Set[int]] = None

    for it in range(1, spec.max_rounds + 1):
        round_start = time.time()
        pct = int(100 * it / spec.max_rounds)
        print(f"[Round {it}/{spec.max_rounds} | ~{pct}%] Scope size: {len(scope)}")

        debug_scope_snapshot(provider, scope)
        milp = SubnetworkMILP(provider, spec, scope)
        sol = milp.solve()

        if sol.get("status") == "NO_X_CONSUMER_IN_SCOPE":
            print("  -> No reaction in scope consumes X. Expanding scope.")
        elif not sol.get("solved", False):
            debug_infeasibility_hints(provider, scope, spec)
            print(f"  -> MILP infeasible or no solution (time {sol.get('time_s', 0):.2f}s).")
        else:
            dom = dominance_metrics(provider, scope, sol)
            score = network_score(sol, dom, lambda_size=spec.lambda_size)
            k = int(np.sum(sol["y"]))
            print(f"  -> SOLVED in {sol['time_s']:.2f}s | selected={k}, gainX={sol['gainX']:.3f}, "
                  f"min_dom_ratio={dom['dominance_min_ratio']:.2f}, avg_dom_ratio={dom['dominance_avg_ratio']:.2f}, score={score:.3f}")

            if (best is None) or (score > best[0]):
                best = (score, it, len(scope), sol, dom, list(scope))
                hint_species = set(sol["active_species"]) | set(sol["consumed_species"])

        if scope_mgr.is_full():
            print("  -> Scope covers the full network. Stopping expansion.")
            break

        prev = len(scope)
        scope = scope_mgr.expand(hint_species=hint_species)
        print(f"  -> Expanded scope: {prev} -> {len(scope)} | Round time: {time.time()-round_start:.2f}s")

    print("--------------------------------------------------")
    if best is None or not best[3].get("solved", False):
        print("No feasible subnetwork found across rounds under current settings.")
        return

    score, it_best, scope_size, sol_best, dom_best, scope_idx = best
    y = sol_best["y"]
    v = sol_best["v"]
    selected = [j for j, yy in enumerate(y) if yy > 0]
    print("== Best subnetwork summary ==")
    print(f"Found at round {it_best} with scope size {scope_size}.")
    print(f"Selected reactions: {len(selected)}")
    print(f"Active species: {len(sol_best['active_species'])}, Consumed species: {len(sol_best['consumed_species'])}")
    print(f"Net gain of X (species {X}): {sol_best['gainX']:.3f}")
    print(f"Dominance min ratio: {dom_best['dominance_min_ratio']:.3f}, avg ratio: {dom_best['dominance_avg_ratio']:.3f}")
    print(f"Final score: {score:.3f}")

    nonzero_v = [(j, float(vj)) for j, vj in enumerate(v) if vj > 1e-9]
    nonzero_v = sorted(nonzero_v, key=lambda t: -t[1])
    print("Top flux reactions (local indices within scope):")
    for j, val in nonzero_v[:20]:
        print(f"  r{j}: v={val:.3f}")


# ------------------------------ Main ------------------------------

if __name__ == "__main__":
    # Problem-specific knobs (keep solver generic)
    spec = ProblemSpec(
        eps_gain_X=1.0,
        eps_active=1e-3,
        lambda_size=1e-2,
        target_min_demand=0.0,   # set >0 to enforce minimal net production per target species
        target_weight=0.5,       # objective weight for targets
        enable_solver_log=False, # set True to see CBC logs
        max_rounds=6,
        init_scope=120,
        scope_step=80,
    )

    from real_chem_provider import RealChemProvider
    provider = RealChemProvider(base_dir=".")
    provider.build_reactions_from_templates(max_reactions=2000)

    search_best_network(provider, spec, random_seed=42)
