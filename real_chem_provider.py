# real_chem_provider.py
# RealChemProvider: reads your CSV/JSON/YAML data and exposes the ChemProvider API.

from __future__ import annotations
from typing import List, Set, Dict, Optional
import os
import math
import json
import numpy as np
import pandas as pd
import yaml

# RDKit is used only to convert SMILES -> InChI (for KEGG matching)
try:
    from rdkit import Chem
    from rdkit.Chem import inchi as rd_inchi
except Exception as e:
    raise RuntimeError("Please install RDKit (pip install rdkit-pypi).") from e


class RealChemProvider:
    """
    Minimal, file-backed ChemProvider:
      - species = FOOD ? TARGET ? {H2O}
      - ?fG' computed from eQuilibrator pseudoisomers (approximate transform vs pH)
      - reaction set: initially empty (templates are loaded but not applied here)
      - all getters conform to the solver expectations
    You can later extend _build_reactions_from_templates() to generate real reactions.
    """

    # ----------------- configuration -----------------
    def __init__(
        self,
        base_dir: str = ".",
        food_csv: str = "data/food.csv",
        targets_csv: str = "data/targets.csv",
        env_yaml: str = "config/env.yaml",
        kegg_compounds_json: str = "external/kegg_compounds.json",
        kegg_pseudoisomers_csv: str = "external/kegg_pseudoisomers_Alberty.csv",
        templates_yaml: str = "templates/aqueous_min.yaml",
        alpha_dominance: float = 5.0,
        max_selected: Optional[int] = 60,
        default_food_budget: float = 50.0,
        seed: int = 0,
    ):
        self.rng = np.random.default_rng(seed)
        self._alpha = float(alpha_dominance)
        self._max_selected = max_selected

        self.paths = {
            "food_csv": os.path.join(base_dir, food_csv),
            "targets_csv": os.path.join(base_dir, targets_csv),
            "env_yaml": os.path.join(base_dir, env_yaml),
            "kegg_compounds_json": os.path.join(base_dir, kegg_compounds_json),
            "kegg_pseudoisomers_csv": os.path.join(base_dir, kegg_pseudoisomers_csv),
            "templates_yaml": os.path.join(base_dir, templates_yaml),
        }

        self.env = self._load_env(self.paths["env_yaml"])
        self.T = float(self.env.get("temperature_K", 298.15))
        self.pH = float(self.env.get("pH", 7.0))
        self.I = float(self.env.get("ionic_strength_M", 0.1))
        self.R = 8.314462618e-3  # kJ/mol/K

        self.templates = self._load_templates(self.paths["templates_yaml"])
        self.kegg_index = self._index_kegg_json(self.paths["kegg_compounds_json"])
        self.pseudo_table = self._load_pseudoisomers(self.paths["kegg_pseudoisomers_csv"])

        species_rows = self._load_species(self.paths["food_csv"], self.paths["targets_csv"])
        self._species_table = self._finalize_species(species_rows)
        self._m = len(self._species_table)

        self._food_indices = {i for i, r in enumerate(self._species_table) if r["is_food"]}
        self._target_indices = {i for i, r in enumerate(self._species_table) if r["is_target"]}

        # pick X: prefer a non-food target, else any non-food, else 0
        self._X = self._choose_autocatalyst_index()

        # budgets
        self._food_budget = np.zeros(self._m, dtype=float)
        for i in self._food_indices:
            self._food_budget[i] = float(self.env.get("food_budget", default_food_budget))

        # reactions (initially none; extend later)
        self._S = np.zeros((self._m, 0), dtype=float)
        self._U = np.zeros(0, dtype=float)
        self._delta_rG = np.zeros(0, dtype=float)
        self._allowed_global = np.zeros(0, dtype=bool)
        self._participants_all: List[List[int]] = []
        self._usesX_all = np.zeros(0, dtype=bool)

        # precompute consumes
        self._consumes = np.maximum(-self._S, 0.0)

    # ----------------- I/O helpers -----------------
    @staticmethod
    def _load_env(path: str) -> Dict[str, float]:
        if not os.path.isfile(path):
            return {"temperature_K": 298.15, "pH": 7.0, "ionic_strength_M": 0.1}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _load_templates(path: str) -> Dict:
        if not os.path.isfile(path):
            return {"version": "0.0", "templates": [], "group_library": {}}
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @staticmethod
    def _index_kegg_json(path: str) -> Dict[str, str]:
        if not os.path.isfile(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        idx = {}
        for row in data:
            cid = row.get("CID")
            inchi = row.get("InChI")
            if cid and inchi:
                idx[inchi.strip()] = cid.strip()
        return idx

    @staticmethod
    def _load_pseudoisomers(path: str) -> pd.DataFrame:
        if not os.path.isfile(path):
            return pd.DataFrame(columns=["cid", "name", "dg_chem_kjmol", "nH", "charge", "nMg", "note"])
        try:
            df = pd.read_csv(path)
            cols = [c.lower() for c in df.columns]
            # try to normalize column names
            rename = {}
            for c in df.columns:
                cl = c.lower()
                if "kegg" in cl or cl == "cid":
                    rename[c] = "cid"
                elif "name" in cl:
                    rename[c] = "name"
                elif "chem" in cl or "formation" in cl:
                    rename[c] = "dg_chem_kjmol"
                elif cl in ("n_h", "nh", "num_h", "h"):
                    rename[c] = "nH"
                elif "charge" in cl:
                    rename[c] = "charge"
                elif "mg" in cl:
                    rename[c] = "nMg"
                elif "note" in cl or "remark" in cl:
                    rename[c] = "note"
            df = df.rename(columns=rename)
        except Exception:
            # fallback for no header
            df = pd.read_csv(path, header=None)
            df.columns = ["cid", "name", "dg_chem_kjmol", "nH", "charge", "nMg", "note"]
        # ensure types
        for col in ["dg_chem_kjmol", "nH", "charge", "nMg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "cid" in df.columns:
            df["cid"] = df["cid"].astype(str).str.strip()
        return df

    def _load_species(self, food_csv: str, targets_csv: str) -> List[Dict]:
        def read_csv(path: str) -> pd.DataFrame:
            return pd.read_csv(path) if os.path.isfile(path) else pd.DataFrame(columns=["id", "name", "smiles"])

        foods = read_csv(food_csv)
        targs = read_csv(targets_csv)

        rows: List[Dict] = []
        for _, r in foods.iterrows():
            rows.append({"id": str(r.get("id", r.get("name", "food"))),
                         "name": str(r.get("name", r.get("id", "food"))),
                         "smiles": str(r.get("smiles", "")).strip(),
                         "is_food": True, "is_target": False})
        for _, r in targs.iterrows():
            rows.append({"id": str(r.get("id", r.get("name", "target"))),
                         "name": str(r.get("name", r.get("id", "target"))),
                         "smiles": str(r.get("smiles", "")).strip(),
                         "is_food": False, "is_target": True})
        # ensure water in pool
        rows.append({"id": "H2O", "name": "water", "smiles": "O", "is_food": False, "is_target": False})
        # deduplicate by smiles
        seen = set()
        uniq = []
        for r in rows:
            key = r["smiles"]
            if key and key not in seen:
                uniq.append(r)
                seen.add(key)
        return uniq

    def _finalize_species(self, rows: List[Dict]) -> List[Dict]:
        out = []
        for r in rows:
            smi = r["smiles"]
            mol = Chem.MolFromSmiles(smi) if smi else None
            inchi = rd_inchi.MolToInchi(mol) if mol is not None else None
            cid = self.kegg_index.get(inchi, None) if inchi else None
            dgf = self._dgfprime_from_pseudoisomers(cid) if cid else None
            out.append({
                "id": r["id"], "name": r["name"], "smiles": smi,
                "inchi": inchi, "kegg_cid": cid, "dgfprime_kjmol": dgf,
                "is_food": bool(r["is_food"]), "is_target": bool(r["is_target"])
            })
        return out

    # ----------------- thermo transform (approximate) -----------------
    def _dgfprime_from_pseudoisomers(self, cid: Optional[str]) -> Optional[float]:
        if not cid or self.pseudo_table.empty:
            return None
        df = self.pseudo_table[self.pseudo_table["cid"].astype(str) == cid]
        if df.empty or "dg_chem_kjmol" not in df.columns or "nH" not in df.columns:
            return None
        # Approximate transformed ?fG'° at given pH: min over pseudoisomers of (?G_chem + nH * RT ln(10) * pH)
        # Ionic strength and Mg terms are ignored here.
        term = self.R * self.T * math.log(10.0) * self.pH
        vals = df["dg_chem_kjmol"].astype(float).values + df["nH"].astype(float).values * term
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(np.min(vals))

    # ----------------- API: sizes and sets -----------------
    def species_count(self) -> int:
        return self._m

    def reaction_count(self) -> int:
        return self._S.shape[1]

    def food_set(self) -> Set[int]:
        return set(self._food_indices)

    def autocatalyst_index(self) -> int:
        return int(self._X)

    def target_species(self) -> Set[int]:
        return set(self._target_indices)

    def alpha_dominance(self) -> float:
        return self._alpha

    def max_selected_reactions(self) -> Optional[int]:
        return self._max_selected

    def food_budget(self) -> np.ndarray:
        return self._food_budget.copy()

    # ----------------- API: matrices for a scope -----------------
    def stoichiometry_submatrix(self, scope: List[int]) -> np.ndarray:
        if len(scope) == 0:
            return np.zeros((self._m, 0), dtype=float)
        return self._S[:, scope]

    def capacity_upper_bounds(self, scope: List[int]) -> np.ndarray:
        if len(scope) == 0:
            return np.zeros(0, dtype=float)
        return self._U[scope]

    def thermo_allowed_mask(self, scope: List[int]) -> np.ndarray:
        if len(scope) == 0:
            return np.zeros(0, dtype=bool)
        return self._allowed_global[scope]

    def consumes_submatrix(self, scope: List[int]) -> np.ndarray:
        if len(scope) == 0:
            return np.zeros((self._m, 0), dtype=float)
        return self._consumes[:, scope]

    def participants_lists(self, scope: List[int]) -> List[List[int]]:
        return [self._participants_all[j] for j in scope] if scope else []

    def uses_X_mask(self, scope: List[int]) -> np.ndarray:
        if len(scope) == 0:
            return np.zeros(0, dtype=bool)
        return self._usesX_all[scope]

    # ----------------- API: helpers for scope -----------------
    def global_allowed_mask(self) -> np.ndarray:
        return self._allowed_global.copy()

    def reaction_participants_all(self, j: int) -> List[int]:
        return self._participants_all[j]

    def delta_rG_all(self) -> np.ndarray:
        return self._delta_rG.copy()

    # ----------------- optional: template application (stub) -----------------
    # Extend this method to populate self._S, self._U, self._delta_rG, self._allowed_global, self._participants_all, self._usesX_all
    def build_reactions_from_templates(self, max_reactions: int = 0) -> None:
        # Placeholder: no reactions are generated here.
        # Keep shapes consistent.
        self._S = np.zeros((self._m, 0), dtype=float)
        self._U = np.zeros(0, dtype=float)
        self._delta_rG = np.zeros(0, dtype=float)
        self._allowed_global = np.zeros(0, dtype=bool)
        self._participants_all = []
        self._usesX_all = np.zeros(0, dtype=bool)
        self._consumes = np.maximum(-self._S, 0.0)

    # ----------------- internal -----------------
    def _choose_autocatalyst_index(self) -> int:
        # prefer first target if any, else first non-food
        for i, r in enumerate(self._species_table):
            if r["is_target"]:
                return i
        for i, r in enumerate(self._species_table):
            if not r["is_food"]:
                return i
        return 0
