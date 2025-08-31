# real_chem_provider.py
# RealChemProvider: file-backed ChemProvider with template expansion (no RDKit required).

from __future__ import annotations
from typing import List, Set, Dict, Optional, Tuple
import os, math, json
import numpy as np
import pandas as pd
import yaml
from rdkit import Chem
from rdkit.Chem import rdChemReactions

class RealChemProvider:
    """
    File-backed ChemProvider:
      - Species pool = FOOD ? TARGET ?
      - Groups for template matching come from a 'groups' column (comma/semicolon-separated tags).
      - KEGG CID resolved from data/kegg_map.csv (optional). If missing, ?fG' may be None.
      - ?fG' computed from eQuilibrator pseudoisomers (SBtab TSV) via a simple pH transform.
      - Reactions are generated from templates by group tags (no SMARTS, no RDKit).
    """

    def __init__(
        self,
        base_dir: str = ".",
        food_csv: str = "data/food_test.csv",
        targets_csv: str = "data/targets_test.csv",
        kegg_map_csv: str = "data/kegg_map.csv",
        env_yaml: str = "config/env.yaml",
        kegg_compounds_json: str = "external/kegg_compounds.json",  # unused here
        kegg_pseudoisomers_csv: str = "external/kegg_pseudoisomers_Alberty.csv",
        alpha_dominance: float = 5.0,
        max_selected: Optional[int] = 60,
        default_food_budget: float = 50.0,
        seed: int = 0,
    ):
        self._alpha = float(alpha_dominance)
        self._max_selected = max_selected
        self.rng = np.random.default_rng(seed)

        self.paths = {
            "food_csv": os.path.join(base_dir, food_csv),
            "targets_csv": os.path.join(base_dir, targets_csv),
            "kegg_map_csv": os.path.join(base_dir, kegg_map_csv),
            "env_yaml": os.path.join(base_dir, env_yaml),
            "kegg_pseudoisomers_csv": os.path.join(base_dir, kegg_pseudoisomers_csv),
        }

        self.env = self._load_env(self.paths["env_yaml"])
        self.T = float(self.env.get("temperature_K", 298.15))
        self.pH = float(self.env.get("pH", 7.0))
        self.I = float(self.env.get("ionic_strength_M", 0.1))
        self.R = 8.314462618e-3  # kJ/mol/K

        # Engine: "smirks" = explicit rules; "rdkit" = template expansion
        self.engine = str(self.env.get("engine", "smirks")).lower().strip()
        self.max_depth = int(self.env.get("max_depth", 1))  # not used in this light path
        self.smirks_rules_path = str(self.env.get("smirks_rules", "templates/smirks_rules.yaml"))

        # Load rule file (used by both explicit and template modes)
        self.smirks_rules = self._load_templates(os.path.join(base_dir, self.smirks_rules_path))

        self.pseudo_table = self._load_pseudoisomers(self.paths["kegg_pseudoisomers_csv"])
        self.map_by_smiles, self.map_by_name = self._load_kegg_map(self.paths["kegg_map_csv"])

        species_rows = self._load_species(self.paths["food_csv"], self.paths["targets_csv"])
        self._species_table = self._finalize_species(species_rows)
        self._m = len(self._species_table)

        self._food_indices = {i for i, r in enumerate(self._species_table) if r["is_food"]}
        self._target_indices = {i for i, r in enumerate(self._species_table) if r["is_target"]}

        self._X = self._choose_autocatalyst_index()

        self._food_budget = np.zeros(self._m, dtype=float)
        for i in self._food_indices:
            self._food_budget[i] = float(self.env.get("food_budget", default_food_budget))

        # Empty reaction matrices (to be filled later)
        self._S = np.zeros((self._m, 0), dtype=float)
        self._U = np.zeros(0, dtype=float)
        self._delta_rG = np.zeros(0, dtype=float)
        self._allowed_global = np.zeros(0, dtype=bool)
        self._participants_all: List[List[int]] = []
        self._usesX_all = np.zeros(0, dtype=bool)
        self._consumes = np.maximum(-self._S, 0.0)



    # -------------------- I/O helpers --------------------
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
    def _load_csv_if_exists(path: str, req_cols: Tuple[str, ...]) -> pd.DataFrame:
        if not os.path.isfile(path):
            return pd.DataFrame(columns=list(req_cols))
        df = pd.read_csv(path)
        for c in req_cols:
            if c not in df.columns:
                df[c] = ""
        return df

    def _load_kegg_map(self, path: str) -> Tuple[Dict[str, str], Dict[str, str]]:
        df = self._load_csv_if_exists(path, ("name", "smiles", "kegg_cid"))
        by_smi, by_name = {}, {}
        for _, r in df.iterrows():
            smi = str(r.get("smiles", "")).strip()
            name = str(r.get("name", "")).strip()
            cid = str(r.get("kegg_cid", "")).strip()
            if cid:
                if smi:
                    by_smi[smi] = cid
                if name:
                    by_name[name] = cid
        return by_smi, by_name

    def _load_species(self, food_csv: str, targets_csv: str) -> List[Dict]:
        def read_csv(path: str) -> pd.DataFrame:
            return self._load_csv_if_exists(path, ("id", "name", "smiles"))

        foods = read_csv(food_csv); foods["is_food"] = True;  foods["is_target"] = False
        targs = read_csv(targets_csv); targs["is_food"] = False; targs["is_target"] = True

        df = pd.concat([foods, targs], ignore_index=True)

        # Ensure water present (non-food by default; make it food in CSV if you need buffered H2O)
        water = pd.DataFrame([{"id": "H2O", "name": "water", "smiles": "O", "is_food": False, "is_target": False}])
        df = pd.concat([df, water], ignore_index=True)

        # Deduplicate by SMILES (keep first occurrence)
        seen = set(); rows: List[Dict] = []
        for _, r in df.iterrows():
            smi = str(r.get("smiles", "")).strip()
            if not smi or smi in seen:
                continue
            seen.add(smi)
            rows.append({
                "id": str(r.get("id", r.get("name", ""))),
                "name": str(r.get("name", "")),
                "smiles": smi,
                "is_food": bool(r.get("is_food", False)),
                "is_target": bool(r.get("is_target", False)),
            })
        return rows


    def _finalize_species(self, rows: List[Dict]) -> List[Dict]:
        out: List[Dict] = []
        for r in rows:
            smi = r["smiles"]; name = r["name"]
            cid = self.map_by_smiles.get(smi) or self.map_by_name.get(name)
            dgf = self._dgfprime_from_pseudoisomers(cid) if cid else None
            out.append({
                "id": r["id"], "name": name, "smiles": smi,
                "kegg_cid": cid, "dgfprime_kjmol": dgf,
                "is_food": bool(r["is_food"]), "is_target": bool(r["is_target"]),
            })
        return out


    # -------------------- smirks helpers --------------------
    def _canonical_smiles(self, smi: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return smi.strip()
            return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        except Exception:
            return smi.strip()

    def _mol_from_smiles(self, smi: str):
        try:
            return Chem.MolFromSmiles(smi)
        except Exception:
            return None

    def _add_species_from_smiles(self, smi: str, name_hint: Optional[str] = None) -> int:
        csmi = self._canonical_smiles(smi)
        for idx, r in enumerate(self._species_table):
            if r.get("smiles", "") == csmi:
                return idx
        entry = {
            "id": f"AUTO_{len(self._species_table)}",
            "name": name_hint or csmi,
            "smiles": csmi,
            "kegg_cid": None,
            "dgfprime_kjmol": None,
            "is_food": False,
            "is_target": False,
        }
        self._species_table.append(entry)
        self._m = len(self._species_table)
        if self._food_budget.shape[0] < self._m:
            pad = self._m - self._food_budget.shape[0]
            self._food_budget = np.pad(self._food_budget, (0, pad))
        return self._m - 1


    # -------------------- pseudoisomers (SBtab) --------------------
    @staticmethod
    def _load_pseudoisomers(path: str) -> pd.DataFrame:
        import io
        if not os.path.isfile(path):
            return pd.DataFrame(columns=["cid","name","dg_chem_kjmol","nH","charge","nMg","note"])
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().replace("\r\n", "\n").replace("\r", "\n")
        lines = raw.split("\n")
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith("!") and "Identifiers:kegg.compound" in line:
                header_idx = i; break
        if header_idx is not None:
            tsv = "\n".join(lines[header_idx:])
            df = pd.read_csv(io.StringIO(tsv), sep="\t")
            rename = {}
            for c in df.columns:
                cl = str(c).strip()
                if cl == "!Identifiers:kegg.compound": rename[c] = "cid"
                elif cl == "!Name": rename[c] = "name"
                elif cl == "!Mean": rename[c] = "dg_chem_kjmol"
                elif cl == "!nH": rename[c] = "nH"
                elif cl == "!Charge": rename[c] = "charge"
                elif cl == "!nMg": rename[c] = "nMg"
                elif cl == "!Comment": rename[c] = "note"
            df = df.rename(columns=rename)
        else:
            # Fallback: try as CSV/TSV with various seps
            df = None
            for sep in [",", "\t", ";", "|"]:
                try:
                    tmp = pd.read_csv(path, sep=sep)
                    df = tmp; break
                except Exception:
                    pass
            if df is None:
                return pd.DataFrame(columns=["cid","name","dg_chem_kjmol","nH","charge","nMg","note"])
            # Heuristic rename
            rename = {}
            for c in df.columns:
                cl = str(c).strip().lower()
                if "kegg" in cl or cl in ("cid","compound id"): rename[c] = "cid"
                elif "name" in cl: rename[c] = "name"
                elif any(k in cl for k in ["mean","gibbs","dg","formation","chem"]): rename[c] = "dg_chem_kjmol"
                elif cl in ("nh","n_h","num_h","h","n h"): rename[c] = "nH"
                elif "charge" in cl: rename[c] = "charge"
                elif "mg" in cl: rename[c] = "nMg"
                elif any(k in cl for k in ["note","comment","remark"]): rename[c] = "note"
            df = df.rename(columns=rename)

        for col in ["dg_chem_kjmol","nH","charge","nMg"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "cid" in df.columns:
            df["cid"] = df["cid"].astype(str).str.strip()
        # Keep only expected columns
        for col in ["cid","name","dg_chem_kjmol","nH","charge","nMg","note"]:
            if col not in df.columns: df[col] = np.nan
        return df[["cid","name","dg_chem_kjmol","nH","charge","nMg","note"]]

    # -------------------- ?fG' transform --------------------
    def _dgfprime_from_pseudoisomers(self, cid: Optional[str]) -> Optional[float]:
        if not cid or self.pseudo_table.empty:
            return None
        df = self.pseudo_table[self.pseudo_table["cid"].astype(str) == str(cid)]
        if df.empty or "dg_chem_kjmol" not in df.columns or "nH" not in df.columns:
            return None
        term = self.R * self.T * math.log(10.0) * self.pH
        vals = df["dg_chem_kjmol"].astype(float).values + df["nH"].astype(float).values * term
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return None
        return float(np.min(vals))

    # -------------------- ChemProvider API --------------------
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

    def stoichiometry_submatrix(self, scope: List[int]) -> np.ndarray:
        return self._S[:, scope] if scope else np.zeros((self._m, 0), dtype=float)

    def capacity_upper_bounds(self, scope: List[int]) -> np.ndarray:
        return self._U[scope] if scope else np.zeros(0, dtype=float)

    def thermo_allowed_mask(self, scope: List[int]) -> np.ndarray:
        return self._allowed_global[scope] if scope else np.zeros(0, dtype=bool)

    def consumes_submatrix(self, scope: List[int]) -> np.ndarray:
        return self._consumes[:, scope] if scope else np.zeros((self._m, 0), dtype=float)

    def participants_lists(self, scope: List[int]) -> List[List[int]]:
        return [self._participants_all[j] for j in scope] if scope else []

    def uses_X_mask(self, scope: List[int]) -> np.ndarray:
        return self._usesX_all[scope] if scope else np.zeros(0, dtype=bool)

    def global_allowed_mask(self) -> np.ndarray:
        return self._allowed_global.copy()

    def reaction_participants_all(self, j: int) -> List[int]:
        return self._participants_all[j]

    def delta_rG_all(self) -> np.ndarray:
        return self._delta_rG.copy()

    # -------------------- Reaction generation from SMILES rules (feed-forward) --------------------
    def build_reactions_from_smirks(self, max_reactions: int = 5000) -> None:
        """
        Feed-forward expansion using a rule file (config/env.yaml -> smirks_rules).
        Supports rules that give explicit reactant/product SMILES:
          - id: ...
            reversible: false
            reactants_smiles: ["SMI_A", "SMI_B"]   # 1 or 2 reactants supported
            products_smiles:  ["SMI_C", "SMI_D"]
        Unknown species are auto-added as non-food/non-target.
        """
        if self.engine != "smirks":
            print("[WARN] Engine is not 'smirks'; no reactions will be built.")
            self._S = np.zeros((self._m, 0), dtype=float)
            self._U = np.zeros(0, dtype=float)
            self._delta_rG = np.zeros(0, dtype=float)
            self._allowed_global = np.zeros(0, dtype=bool)
            self._participants_all = []
            self._usesX_all = np.zeros(0, dtype=bool)
            self._consumes = np.maximum(-self._S, 0.0)
            return

        rules = self.smirks_rules.get("rules", [])
        if not isinstance(rules, list):
            print("[WARN] smirks_rules.yaml has no 'rules' list; no reactions built.")
            rules = []

        # Fast lookup for species by canonical SMILES
        smi_to_idx: Dict[str, int] = {}
        for idx, r in enumerate(self._species_table):
            smi_to_idx[self._canonical_smiles(r["smiles"])] = idx

        def ensure_species(smi: str, name_hint: Optional[str] = None) -> int:
            csmi = self._canonical_smiles(smi)
            if csmi in smi_to_idx:
                return smi_to_idx[csmi]
            new_idx = self._add_species_from_smiles(csmi, name_hint=name_hint)
            smi_to_idx[csmi] = new_idx
            return new_idx

        # First pass: only collect (reactants_idx, products_idx) pairs,
        # so we can add species freely and build columns later with final m.
        rxn_pairs: List[Tuple[List[int], List[int]]] = []

        for rule in rules:
            rsmis = rule.get("reactants_smiles") or rule.get("reactants") or []
            psmis = rule.get("products_smiles") or rule.get("products") or []
            reversible = bool(rule.get("reversible", False))

            if not rsmis or not psmis:
                sm = rule.get("smirks", None)
                if sm:
                    print(f"[INFO] Skipping SMIRKS-only rule '{rule.get('id','?')}' (no explicit reactant/product SMILES).")
                continue

            rsmis_c = [self._canonical_smiles(s) for s in rsmis]
            psmis_c = [self._canonical_smiles(s) for s in psmis]

            # Make sure product species exist
            prod_idx_vec = [ensure_species(s) for s in psmis_c]

            if len(rsmis_c) == 1:
                i0 = ensure_species(rsmis_c[0])
                rxn_pairs.append(([i0], prod_idx_vec))
                if reversible:
                    rxn_pairs.append((prod_idx_vec, [i0]))

            elif len(rsmis_c) == 2:
                i0 = ensure_species(rsmis_c[0])
                i1 = ensure_species(rsmis_c[1])
                if i0 != i1:
                    rxn_pairs.append(([i0, i1], prod_idx_vec))
                    if reversible:
                        rxn_pairs.append((prod_idx_vec, [i0, i1]))
            else:
                print(f"[INFO] Rule '{rule.get('id','?')}' has arity {len(rsmis_c)}; skipping.")
                continue

            if len(rxn_pairs) >= max_reactions:
                break

        # Second pass: build matrices at the FINAL species count
        m = self._m
        S_cols: List[np.ndarray] = []
        U_list: List[float] = []
        drg_list: List[float] = []
        allowed_list: List[bool] = []
        participants: List[List[int]] = []
        usesX: List[bool] = []

        def drg_of(col_vec: np.ndarray) -> Tuple[float, bool]:
            # ?rG' = sum(nu_i * ?fG'_i) with nu>0 for products, nu<0 for reactants
            dgf = [sp.get("dgfprime_kjmol", None) for sp in self._species_table]
            have_all = True
            delta = 0.0
            for i, nu in enumerate(col_vec):
                if abs(nu) < 1e-12:
                    continue
                if dgf[i] is None:
                    have_all = False
                else:
                    delta += float(nu) * float(dgf[i])
            if not have_all:
                return 0.0, True
            return float(delta), (delta <= 0.0)

        for react_idx, prod_idx in rxn_pairs[:max_reactions]:
            col = np.zeros(m, dtype=float)
            for i in react_idx:
                if 0 <= i < m:
                    col[i] -= 1.0
            for i in prod_idx:
                if 0 <= i < m:
                    col[i] += 1.0
            if np.allclose(col, 0.0):
                continue

            drg, allowed = drg_of(col)
            S_cols.append(col)
            U_list.append(float(self.rng.uniform(1.0, 10.0)))
            drg_list.append(drg)
            allowed_list.append(allowed)
            parts = sorted(list(set(react_idx + prod_idx)))
            participants.append(parts)
            usesX.append(col[self._X] < 0.0)

        if S_cols:
            S = np.stack(S_cols, axis=1)
        else:
            S = np.zeros((m, 0), dtype=float)

        self._S = S
        self._U = np.array(U_list, dtype=float)
        self._delta_rG = np.array(drg_list, dtype=float)
        self._allowed_global = np.array(allowed_list, dtype=bool)
        self._participants_all = participants
        self._usesX_all = np.array(usesX, dtype=bool)
        self._consumes = np.maximum(-self._S, 0.0)

        print(f"[SMIRKS] Built reactions: k={self._S.shape[1]} | species m={self._m}")

    def build_reactions_from_templates_rdkit(
      self,
      max_reactions: int = 5000,
      max_pairs_default: int = 2000,
      max_outcomes_per_pair_default: int = 8,
    ) -> None:
      """
      Template expansion using RDKit reaction SMARTS/SMIRKS from smirks_rules.yaml.
      Each rule must provide `smirks: "..."`.
      We support arity 1 or 2. For each matching reactant (or pair) we generate
      product sets, auto-adding unseen species by SMILES. Stoichiometry is 1:1.
      IMPORTANT: We match & run reactions on molecules with EXPLICIT hydrogens,
      because many templates use [H] explicitly. Products are then stripped of
      explicit H before canonical SMILES.
      """

      if self.engine != "rdkit":
          print("[WARN] Engine is not 'rdkit'; no template expansion performed.")
          # Keep matrices empty
          self._S = np.zeros((self._m, 0), dtype=float)
          self._U = np.zeros(0, dtype=float)
          self._delta_rG = np.zeros(0, dtype=float)
          self._allowed_global = np.zeros(0, dtype=bool)
          self._participants_all = []
          self._usesX_all = np.zeros(0, dtype=bool)
          self._consumes = np.maximum(-self._S, 0.0)
          return

      rules = self.smirks_rules.get("rules", [])
      if not isinstance(rules, list):
          print("[WARN] smirks_rules.yaml has no 'rules' list; no reactions built.")
          rules = []

      # Canonical SMILES map and molecule caches (implicit-H and explicit-H)
      smi_to_idx: Dict[str, int] = {}
      mol_cache_noH: List[Optional[Chem.Mol]] = []
      mol_cache_H: List[Optional[Chem.Mol]] = []
      for idx, r in enumerate(self._species_table):
          csmi = self._canonical_smiles(r["smiles"])
          smi_to_idx[csmi] = idx
          m = self._mol_from_smiles(csmi)
          mol_cache_noH.append(m)
          mol_cache_H.append(Chem.AddHs(m) if m is not None else None)

      def ensure_species(smi: str, name_hint: Optional[str] = None) -> int:
          csmi = self._canonical_smiles(smi)
          idx = smi_to_idx.get(csmi, None)
          if idx is not None:
              return idx
          new_idx = self._add_species_from_smiles(csmi, name_hint=name_hint)
          smi_to_idx[csmi] = new_idx
          m = self._mol_from_smiles(csmi)
          mol_cache_noH.append(m)
          mol_cache_H.append(Chem.AddHs(m) if m is not None else None)
          return new_idx

      # Collect (reactant_idx_list, product_idx_list)
      rxn_pairs: List[Tuple[List[int], List[int]]] = []

      for rule in rules:
          rid = str(rule.get("id", "?"))
          smirks = rule.get("smirks", None)
          if not smirks:
              # Not a template rule; skip (explicit rules handled in build_reactions_from_smirks)
              continue

          try:
              rxn = rdChemReactions.ReactionFromSmarts(smirks, useSmiles=True)
          except Exception as e:
              print(f"[WARN] Could not parse rule '{rid}': {e}")
              continue

          arity = rxn.GetNumReactantTemplates()
          if arity not in (1, 2):
              print(f"[INFO] Rule '{rid}' has arity {arity}; skipping (only 1 or 2 supported).")
              continue

          reversible = bool(rule.get("reversible", False))
          max_pairs = int(rule.get("max_pairs", max_pairs_default))
          max_outcomes_per_pair = int(rule.get("max_outcomes_per_pair", max_outcomes_per_pair_default))

          # Candidate selection: use EXPLICIT-H molecules for substructure matching
          cand_lists: List[List[int]] = []
          for rpos in range(arity):
              tmpl = rxn.GetReactantTemplate(rpos)
              if tmpl is None:
                  cand_lists.append([])
                  continue
              matches = []
              for i, molH in enumerate(mol_cache_H):
                  if molH is not None and molH.HasSubstructMatch(tmpl):
                      matches.append(i)
              cand_lists.append(matches)

          # Debug: counts per rule
          try:
              c_counts = " x ".join(str(len(c)) for c in cand_lists)
              print(f"[RDKIT] Rule '{rid}': arity={arity}, candidates={c_counts}")
          except Exception:
              pass

          if any(len(c) == 0 for c in cand_lists):
              continue

          made_pairs = 0

          if arity == 1:
              for i0 in cand_lists[0]:
                  m0H = mol_cache_H[i0]
                  if m0H is None:
                      continue
                  try:
                      prod_sets = rxn.RunReactants((m0H,))
                  except Exception:
                      continue
                  if not prod_sets:
                      continue

                  for prod_tuple in prod_sets[:max_outcomes_per_pair]:
                      prod_smis: List[str] = []
                      for pm in prod_tuple:
                          try:
                              # strip explicit H for canonicalization
                              pm2 = Chem.RemoveHs(pm)
                              Chem.SanitizeMol(pm2)
                              prod_smis.append(Chem.MolToSmiles(pm2, canonical=True, isomericSmiles=True))
                          except Exception:
                              # fall back: try without RemoveHs
                              try:
                                  prod_smis.append(Chem.MolToSmiles(pm, canonical=True, isomericSmiles=True))
                              except Exception:
                                  continue
                      if not prod_smis:
                          continue
                      prod_idx_vec = [ensure_species(s) for s in prod_smis]
                      rxn_pairs.append(([i0], prod_idx_vec))
                      if reversible:
                          rxn_pairs.append((prod_idx_vec, [i0]))
                      if len(rxn_pairs) >= max_reactions:
                          break
                  if len(rxn_pairs) >= max_reactions:
                      break

          else:  # arity == 2
              A, B = cand_lists
              for i0 in A:
                  if len(rxn_pairs) >= max_reactions or made_pairs >= max_pairs:
                      break
                  m0H = mol_cache_H[i0]
                  if m0H is None:
                      continue
                  for i1 in B:
                      if len(rxn_pairs) >= max_reactions or made_pairs >= max_pairs:
                          break
                      if i1 == i0:
                          continue
                      m1H = mol_cache_H[i1]
                      if m1H is None:
                          continue
                      try:
                          prod_sets = rxn.RunReactants((m0H, m1H))
                      except Exception:
                          continue
                      if not prod_sets:
                          continue

                      made_pairs += 1
                      for prod_tuple in prod_sets[:max_outcomes_per_pair]:
                          prod_smis: List[str] = []
                          for pm in prod_tuple:
                              try:
                                  pm2 = Chem.RemoveHs(pm)
                                  Chem.SanitizeMol(pm2)
                                  prod_smis.append(Chem.MolToSmiles(pm2, canonical=True, isomericSmiles=True))
                              except Exception:
                                  try:
                                      prod_smis.append(Chem.MolToSmiles(pm, canonical=True, isomericSmiles=True))
                                  except Exception:
                                      continue
                          if not prod_smis:
                              continue
                          prod_idx_vec = [ensure_species(s) for s in prod_smis]
                          rxn_pairs.append(([i0, i1], prod_idx_vec))
                          if reversible:
                              rxn_pairs.append((prod_idx_vec, [i0, i1]))
                          if len(rxn_pairs) >= max_reactions:
                              break

      # Build matrices (final m)
      m = self._m
      S_cols: List[np.ndarray] = []
      U_list: List[float] = []
      drg_list: List[float] = []
      allowed_list: List[bool] = []
      participants: List[List[int]] = []
      usesX: List[bool] = []

      def drg_of(col_vec: np.ndarray) -> Tuple[float, bool]:
          dgf = [sp.get("dgfprime_kjmol", None) for sp in self._species_table]
          have_all = True
          delta = 0.0
          for i, nu in enumerate(col_vec):
              if abs(nu) < 1e-12:
                  continue
              if dgf[i] is None:
                  have_all = False
              else:
                  delta += float(nu) * float(dgf[i])
          if not have_all:
              return 0.0, True
          return float(delta), (delta <= 0.0)

      for react_idx, prod_idx in rxn_pairs[:max_reactions]:
          col = np.zeros(m, dtype=float)
          for i in react_idx:
              if 0 <= i < m:
                  col[i] -= 1.0
          for i in prod_idx:
              if 0 <= i < m:
                  col[i] += 1.0
          if np.allclose(col, 0.0):
              continue

          drg, allowed = drg_of(col)
          S_cols.append(col)
          U_list.append(float(self.rng.uniform(1.0, 10.0)))
          drg_list.append(drg)
          allowed_list.append(allowed)
          parts = sorted(list(set(react_idx + prod_idx)))
          participants.append(parts)
          usesX.append(col[self._X] < 0.0)

      if S_cols:
          S = np.stack(S_cols, axis=1)
      else:
          S = np.zeros((m, 0), dtype=float)

      self._S = S
      self._U = np.array(U_list, dtype=float)
      self._delta_rG = np.array(drg_list, dtype=float)
      self._allowed_global = np.array(allowed_list, dtype=bool)
      self._participants_all = participants
      self._usesX_all = np.array(usesX, dtype=bool)
      self._consumes = np.maximum(-self._S, 0.0)

      print(f"[RDKIT] Built reactions: k={self._S.shape[1]} | species m={self._m}")


    # -------------------- internals --------------------
    def _choose_autocatalyst_index(self) -> int:
        for i, r in enumerate(self._species_table):
            if r["is_target"]:
                return i
        for i, r in enumerate(self._species_table):
            if not r["is_food"]:
                return i
        return 0
