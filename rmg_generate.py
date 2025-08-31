# offline_generate_minimal.py
# Minimal offline generator (RMG-Py 3.3.0) that:
# - loads a small whitelist of kinetics families
# - reads a food CSV (id,name,smiles)
# - expands reactions breadth-first up to a small depth
# - applies basic aqueous filters (CHONPS, no radicals, neutral, heavy-atom cap)
# - writes species.csv and reactions_kinetics.csv
#
# Run inside the RMG 3.3.0 container with volumes mounted, e.g.:
#   docker run --rm -it \
#     -e RMG_INPUT_DIR=/rmgdb/input \
#     -v "C:\...\RMG-database\input:/rmgdb/input" \
#     -v "C:\...\ChemicalExploration:/work" \
#     reactionmechanismgenerator/rmg:3.3.0
# Then in the container:
#   python /work/offline_generate_minimal.py \
#     --food-csv /work/food.csv \
#     --out-dir /work/out

import os
import os.path as op
import csv
import math
import argparse
from collections import deque

from rmgpy.data.thermo import ThermoDatabase
from rmgpy.data.kinetics import KineticsDatabase
from rmgpy.species import Species
from rmgpy.molecule import translator as tr
from rmgpy.kinetics.arrhenius import Arrhenius

R_KJMOLK = 8.314462618e-3  # kJ/mol/K

def parse_args():
    ap = argparse.ArgumentParser(description="Offline reaction dump (minimal).")
    ap.add_argument("--rmg-input-dir", default=os.environ.get("RMG_INPUT_DIR", "/rmgdb/input"))
    ap.add_argument("--food-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-species", type=int, default=2000)
    ap.add_argument("--max-reactions", type=int, default=20000)
    ap.add_argument("--heavy-atom-limit", type=int, default=20)
    ap.add_argument("--families", nargs="+", default=["Ketoenol"])  # start small; extend later
    ap.add_argument("--pairs-per-depth", type=int, default=5000)
    return ap.parse_args()

def allowed_elements_ok(mol):
    allowed = {"C", "H", "O", "N", "P", "S"}
    for a in mol.atoms:
        if a.element.symbol not in allowed:
            return False
    return True

def heavy_atoms(mol):
    return sum(1 for a in mol.atoms if a.element.symbol != "H")

def is_radical(mol):
    if hasattr(mol, "getRadicalCount"):
        return mol.getRadicalCount() > 0
    return False

def net_charge(mol):
    if hasattr(mol, "getNetCharge"):
        return mol.getNetCharge()
    if hasattr(mol, "get_net_charge"):
        return mol.get_net_charge()
    if hasattr(mol, "charge"):
        return mol.charge
    return 0

def to_smiles_species(x):
    if hasattr(x, "molecule") and x.molecule:
        return tr.to_smiles(x.molecule[0])
    return tr.to_smiles(x)

def to_smiles_mol(mol):
    return tr.to_smiles(mol)

def species_from_smiles(smiles):
    """
    Build an RMG Species from a string in the 'smiles' column.
    Special-case radicals that SMILES+OpenBabel do not disambiguate well.
    """
    s = Species()
    txt = smiles.strip()
    # Hydrogen radical (multiplicity 2). SMILES "[H]" is ambiguous for the OB path.
    if txt == "[H]":
        s.from_adjacency_list("multiplicity 2\n1 H u1 p0 c0")
        return s
    # Oxygen atom radical, if you ever need it later:
    if txt == "[O]":
        s.from_adjacency_list("multiplicity 3\n1 O u2 p2 c0")
        return s
    # Fallback: normal SMILES import
    s.from_smiles(txt)
    return s

def extract_arrhenius_si(rxn, T, kin_db=None):
    """
    Return Arrhenius params in SI (with units_A adjusted to M^-1 s^-1 for 2nd order).
    If rxn.kinetics is None, try to estimate kinetics from the family's rate rules.
    """
    from rmgpy.kinetics.arrhenius import Arrhenius

    kin = getattr(rxn, "kinetics", None)

    # Try to estimate kinetics if missing
    if kin is None and kin_db is not None:
        fam_label = str(getattr(rxn, "family", ""))
        if fam_label and fam_label in kin_db.families:
            fam = kin_db.families[fam_label]
            try:
                # template = list of most specific group entries for this reaction
                template = fam.get_reaction_template(rxn)
                # reaction-path degeneracy
                degeneracy = fam.calculate_degeneracy(rxn, resonance=True)
                kin_est, _entry = fam.get_kinetics_for_template(
                    template, degeneracy=degeneracy, method="rate rules"
                )
                # attach so downstream code can see it (optional)
                rxn.kinetics = kin_est
                kin = kin_est
            except Exception:
                kin = None

    if kin is None:
        return None

    # Convert to simple Arrhenius if needed
    a = None
    if hasattr(kin, "to_arrhenius"):
        try:
            a = kin.to_arrhenius(T)
        except Exception:
            a = None
    if a is None:
        if isinstance(kin, Arrhenius):
            a = kin
        else:
            return None

    try:
        A_si = float(a.A.value_si)
        n = float(getattr(a.n, "value_si", a.n))
        Ea_kJmol = float(a.Ea.value_si) / 1000.0
        T0_K = float(a.T0.value_si)
    except Exception:
        return None

    degeneracy = float(getattr(rxn, "degeneracy", 1.0))
    order_ = len(rxn.reactants)

    # Adjust units for 2nd order: m^3/mol/s -> M^-1 s^-1
    if order_ == 2:
        A_eff = A_si * 1000.0
        units_A = "M^-1 s^-1"
    else:
        A_eff = A_si
        units_A = "s^-1"

    return {
        "A": A_eff * degeneracy,
        "n": n,
        "Ea_kJmol": Ea_kJmol,
        "T0_K": T0_K,
        "order": order_,
        "degeneracy": degeneracy,
        "units_A": units_A,
    }

def passes_filters(mol, heavy_atom_limit):
    # CHONPS only
    for a in mol.atoms:
        sym = a.element.symbol
        if sym not in {"C", "H", "O", "N", "P", "S"}:
            return False
    # allow radicals for radical families
    # if you want to forbid them again later, re-enable:
    # from rmgpy.molecule.molecule import Molecule
    # if hasattr(mol, "getRadicalCount") and mol.getRadicalCount() > 0:
    #     return False
    # neutral only (keep this for now)
    if hasattr(mol, "getNetCharge"):
        if mol.getNetCharge() != 0:
            return False
    elif hasattr(mol, "get_net_charge"):
        if mol.get_net_charge() != 0:
            return False
    elif hasattr(mol, "charge"):
        if mol.charge != 0:
            return False
    # heavy atom cap
    heavy = 0
    for a in mol.atoms:
        if a.element.symbol != "H":
            heavy += 1
            if heavy > heavy_atom_limit:
                return False
    return True


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    thermo_db = ThermoDatabase()
    thermo_db.load(op.join(args.rmg_input_dir, "thermo"), libraries=None, depository=True, surface=False)

    kin_db = KineticsDatabase()
    kin_db.load_families(
        op.join(args.rmg_input_dir, "kinetics", "families"),
        families=args.families,
        depositories=["training"],
    )

    # Fill gaps in rate rules by averaging up the tree (helps estimation)
    for fam in kin_db.families.values():
        try:
            fam.fill_rules_by_averaging_up(verbose=False)
        except Exception:
            pass

    # seed species from food.csv
    species_by_smiles = {}
    food_flags = {}
    frontier = deque()

    with open(args.food_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            smiles = row["smiles"].strip()
            spc = species_from_smiles(smiles)
            mol = spc.molecule[0]
            if not passes_filters(mol, args.heavy_atom_limit):
                continue
            can = to_smiles_mol(mol)
            if can not in species_by_smiles:
                species_by_smiles[can] = spc
                food_flags[can] = True
                frontier.append(can)

    # BFS expansion
    reactions = []
    rxn_keys = set()
    depth = 0
    while frontier and depth < args.max_depth and len(species_by_smiles) < args.max_species and len(reactions) < args.max_reactions:
        this_layer = list(frontier)
        frontier.clear()

        # unimolecular
        for s_can in this_layer:
            spc = species_by_smiles[s_can]
            rxns = kin_db.generate_reactions_from_families(
                reactants=[spc],
                products=None,
                only_families=args.families,
                resonance=True,
            )
            for rxn in rxns:
                fam = str(getattr(rxn, "family", ""))
                r_sm = sorted(to_smiles_species(x) for x in rxn.reactants)
                p_sm = sorted(to_smiles_species(x) for x in rxn.products)
                key = (fam, tuple(r_sm), tuple(p_sm))
                if key in rxn_keys:
                    continue
                kin = extract_arrhenius_si(rxn, args.temperature, kin_db=kin_db)
                if kin is None:
                    continue
                rxn_keys.add(key)
                reactions.append((fam, r_sm, p_sm, kin))
                # add new products to species set
                for sp in rxn.products:
                    sp_can = to_smiles_species(sp)
                    if sp_can not in species_by_smiles:
                        # pick first molecule regardless of resonance for filtering
                        mol = sp.molecule[0] if hasattr(sp, "molecule") and sp.molecule else sp
                        if passes_filters(mol, args.heavy_atom_limit):
                            species_by_smiles[sp_can] = sp if hasattr(sp, "molecule") else species_from_smiles(sp_can)
                            food_flags.setdefault(sp_can, False)
                            frontier.append(sp_can)
                if len(reactions) >= args.max_reactions or len(species_by_smiles) >= args.max_species:
                    break
            if len(reactions) >= args.max_reactions or len(species_by_smiles) >= args.max_species:
                break

        if len(reactions) >= args.max_reactions or len(species_by_smiles) >= args.max_species:
            break

        # bimolecular (limit pairs per depth)
        pairs_done = 0
        all_current = list(species_by_smiles.keys())
        for s_can in this_layer:
            spcA = species_by_smiles[s_can]
            for t_can in all_current:
                spcB = species_by_smiles[t_can]
                rxns = kin_db.generate_reactions_from_families(
                    reactants=[spcA, spcB],
                    products=None,
                    only_families=args.families,
                    resonance=True,
                )
                for rxn in rxns:
                    fam = str(getattr(rxn, "family", ""))
                    r_sm = sorted(to_smiles_species(x) for x in rxn.reactants)
                    p_sm = sorted(to_smiles_species(x) for x in rxn.products)
                    key = (fam, tuple(r_sm), tuple(p_sm))
                    if key in rxn_keys:
                        continue
                    kin = extract_arrhenius_si(rxn, args.temperature, kin_db=kin_db)
                    if kin is None:
                        continue
                    rxn_keys.add(key)
                    reactions.append((fam, r_sm, p_sm, kin))
                    for sp in rxn.products:
                        sp_can = to_smiles_species(sp)
                        if sp_can not in species_by_smiles:
                            mol = sp.molecule[0] if hasattr(sp, "molecule") and sp.molecule else sp
                            if passes_filters(mol, args.heavy_atom_limit):
                                species_by_smiles[sp_can] = sp if hasattr(sp, "molecule") else species_from_smiles(sp_can)
                                food_flags.setdefault(sp_can, False)
                                frontier.append(sp_can)
                    if len(reactions) >= args.max_reactions or len(species_by_smiles) >= args.max_species:
                        break
                pairs_done += 1
                if pairs_done >= args.pairs_per_depth:
                    break
                if len(reactions) >= args.max_reactions or len(species_by_smiles) >= args.max_species:
                    break
            if pairs_done >= args.pairs_per_depth:
                break
            if len(reactions) >= args.max_reactions or len(species_by_smiles) >= args.max_species:
                break

        depth += 1

    # write species.csv
    species_path = op.join(args.out_dir, "species.csv")
    with open(species_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "smiles", "is_food", "is_target"])
        # stable id order by insertion order
        for idx, smi in enumerate(species_by_smiles.keys()):
            spc_id = f"S{idx}"
            name = smi
            is_food = "true" if food_flags.get(smi, False) else "false"
            w.writerow([spc_id, name, smi, is_food, "false"])

    # write reactions_kinetics.csv
    reactions_path = op.join(args.out_dir, "reactions_kinetics.csv")
    with open(reactions_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rxn_id", "family", "reactants_smiles", "products_smiles",
            "order", "degeneracy", "A", "n", "Ea_kJmol", "T0_K", "units_A", "reversible"
        ])
        for i, (fam, r_sm, p_sm, kin) in enumerate(reactions):
            rxn_id = f"R{i}"
            r_join = "|".join(r_sm)
            p_join = "|".join(p_sm)
            w.writerow([
                rxn_id, fam, r_join, p_join,
                int(kin["order"]), float(kin["degeneracy"]),
                f"{kin['A']:.6e}", f"{kin['n']:.6g}",
                f"{kin['Ea_kJmol']:.6g}", f"{kin['T0_K']:.6g}",
                kin["units_A"], "false"
            ])

    print("Wrote:", species_path)
    print("Wrote:", reactions_path)
    print("Species:", len(species_by_smiles), "Reactions:", len(reactions))

if __name__ == "__main__":
    main()
