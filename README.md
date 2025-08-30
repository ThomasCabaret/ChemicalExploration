# ChemicalExploration

Searches for a reaction subnetwork between given “food” species and desired “target” species under constraints (F-generated closure, autocatalysis, simplified kinetic dominance, and basic thermodynamic gating). Math-first MILP core with lightweight, tag-based reaction templates.

## Run
Prereqs (once): `pip install numpy pulp pandas pyyaml`  
Execute: `python proto_meta_search.py`

## How to configure a run (key inputs)
- `data/food.csv` — feed species (min cols: `id,name,smiles,groups`).
- `data/targets.csv` — target species (min cols as above). The main target is used as autocatalyst X.
- `data/intermediates.csv` (optional) — extra species with group tags (e.g., `alcohol`, `thiol`, `amine_primary`) to enable templates.
- `config/env.yaml` — environment (pH, temperature_K, ionic_strength_M, food_budget).
- `templates/aqueous_min.yaml` — reaction templates keyed by group tags (esterification, thioesterification, hydrolyses, etc.).

Logs show rounds, scope size, and `[DBG]` lines to diagnose feasibility (e.g., capacity to produce/consume X).

## Project layout
- `proto_meta_search.py` — entry point; scope expansion, MILP solve, scoring.
- `real_chem_provider.py` — file-backed provider (loads CSV/JSON/YAML; builds S, U, dGr’, masks).
- `templates/` — YAML templates (group-tag rules).
- `data/` — your CSV inputs (food, targets, intermediates).
- `config/` — environment settings.
- `external/` — optional external data:
  - `external/kegg_pseudoisomers_Alberty.csv` — SBtab pseudoisomers for approx. dGf’ (from eQuilibrator downloads).
  - `external/kegg_compounds.json` (optional) — KEGG compound metadata (not required in tag-only mode).
  - `external/RMG-database/` (optional) — RMG templates/kinetics if you later switch to a richer provider.
  - `external/OPERA/` (optional) — predicted properties (logP, logS, pKa) for heuristics.

Notes:
- In tag-only mode the provider does not invent new molecules; ensure species required by templates exist (e.g., include an `alcohol` if you want esterification).
- When dGf’ are missing, reactions may be allowed via placeholders; for meaningful thermodynamics, include the eQuilibrator pseudoisomers file.
