#!/usr/bin/env python3
import os
import os.path as op
import csv
import argparse
from collections import deque, defaultdict

from rmgpy.data.thermo import ThermoDatabase
from rmgpy.data.kinetics import KineticsDatabase
from rmgpy.species import Species
from rmgpy.molecule import translator as tr
from rmgpy.kinetics.arrhenius import Arrhenius

def parse_args():
    ap = argparse.ArgumentParser(description="Offline reaction/species dump via RMG-Py (agnostic CSVs).")
    ap.add_argument("--rmg-input-dir", default=os.environ.get("RMG_INPUT_DIR", "/rmgdb/input"))
    ap.add_argument("--food-csv", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--temperature", type=float, default=298.15)
    ap.add_argument("--max-depth", type=int, default=1)
    ap.add_argument("--max-species", type=int, default=50000)
    ap.add_argument("--max-reactions", type=int, default=500000)
    ap.add_argument("--heavy-atom-limit", type=int, default=20)
    ap.add_argument("--families", nargs="+", default=["Ketoenol"])
    ap.add_argument("--pairs-per-depth", type=int, default=250000)
    ap.add_argument("--allow-radicals", action="store_true", default=True)
    ap.add_argument("--require-neutral", action="store_true", default=True)
    return ap.parse_args()

def print_generation_config(args):
    print("=== Generation config ===")
    print(f"RMG input dir         : {args.rmg_input_dir}")
    print(f"Families (whitelist)  : {', '.join(args.families)}")
    print(f"Reaction orders       : 1, 2")
    print(f"Max depth             : {args.max_depth}")
    print(f"Max species           : {args.max_species}")
    print(f"Max reactions         : {args.max_reactions}")
    print(f"Pairs per depth limit : {args.pairs_per_depth}")
    print("Species filters (pre):")
    print("  Elements            : C,H,O,N,P,S")
    print(f"  Allow radicals      : {bool(args.allow_radicals)}")
    print(f"  Require neutral     : {bool(args.require_neutral)}")
    print(f"  Heavy-atom cap      : {args.heavy_atom_limit}")
    print("Policy (pool): species failing pre-filters are registered for ID consistency but NOT scheduled for exploration.")
    print("Policy (kinetics): reactions without kinetics are KEPT with has_kinetics=false and empty kinetic fields.")

def mol_allowed_elements(mol):
    for a in mol.atoms:
        if a.element.symbol not in {"C", "H", "O", "N", "P", "S"}:
            return False
    return True

def mol_heavy_atoms(mol):
    return sum(1 for a in mol.atoms if a.element.symbol != "H")

def mol_radical_count(mol):
    return mol.getRadicalCount() if hasattr(mol, "getRadicalCount") else 0

def mol_net_charge(mol):
    if hasattr(mol, "getNetCharge"):
        return mol.getNetCharge()
    if hasattr(mol, "get_net_charge"):
        return mol.get_net_charge()
    return getattr(mol, "charge", 0)

def to_smiles_mol(mol):
    return tr.to_smiles(mol)

def to_smiles_species(x):
    if hasattr(x, "molecule") and x.molecule:
        return tr.to_smiles(x.molecule[0])
    return tr.to_smiles(x)

def to_adjlist_mol(mol):
    return mol.to_adjacency_list(remove_h=False)

def species_from_repr(kind, rep):
    s = Species()
    k = (kind or "smiles").strip().lower()
    txt = (rep or "").strip()
    if k == "adjlist":
        adj = txt.replace("\\n", "\n").replace(";", "\n")
        s.from_adjacency_list(adj)
        return s
    s.from_smiles(txt)
    return s

def species_from_row(row):
    if "repr_kind" in row and "repr" in row:
        return species_from_repr(row["repr_kind"], row["repr"])
    elif "smiles" in row:
        return species_from_repr("smiles", row["smiles"])
    else:
        raise ValueError("Row must contain either (repr_kind,repr) or smiles")

def extract_arrhenius_si(rxn, T, kin_db=None):
    kin = getattr(rxn, "kinetics", None)
    if kin is None and kin_db is not None:
        fam_label = str(getattr(rxn, "family", "") or "")
        if fam_label and fam_label in kin_db.families:
            fam = kin_db.families[fam_label]
            try:
                template = fam.get_reaction_template(rxn)
                degeneracy = fam.calculate_degeneracy(rxn, resonance=True)
                kin_est, _entry = fam.get_kinetics_for_template(template, degeneracy=degeneracy, method="rate rules")
                rxn.kinetics = kin_est
                kin = kin_est
            except Exception:
                kin = None
    if kin is None:
        return None
    arr = None
    if hasattr(kin, "to_arrhenius"):
        try:
            arr = kin.to_arrhenius(T)
        except Exception:
            arr = None
    if arr is None:
        if isinstance(kin, Arrhenius):
            arr = kin
        else:
            return None
    try:
        A_si = float(arr.A.value_si)
        n = float(getattr(arr.n, "value_si", arr.n))
        Ea_kJmol = float(arr.Ea.value_si) / 1000.0
        T0_K = float(arr.T0.value_si)
    except Exception:
        return None
    order_ = len(rxn.reactants)
    degeneracy = float(getattr(rxn, "degeneracy", 1.0))
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

def passes_species_filters(mol, heavy_atom_limit, allow_radicals, require_neutral):
    if not mol_allowed_elements(mol):
        return False
    if not allow_radicals and mol_radical_count(mol) > 0:
        return False
    if require_neutral and mol_net_charge(mol) != 0:
        return False
    if mol_heavy_atoms(mol) > heavy_atom_limit:
        return False
    return True

def register_reaction(
    rxn,
    kin_db,
    temperature,
    species_key_to_obj,
    species_key_to_meta,
    frontier,
    heavy_atom_limit,
    allow_radicals,
    require_neutral,
    reactions,
    rxn_keys,
    stats,
    dropped_counts,
):
    def ensure_species_registered(mol, is_food=False, count_as_product=False):
        key = to_adjlist_mol(mol)
        present = key in species_key_to_obj
        if not present:
            spc = Species()
            smi = to_smiles_mol(mol)
            spc.from_smiles(smi)
            species_key_to_obj[key] = spc
            meta = {
                "name": smi,
                "smiles_pretty": smi,
                "charge": mol_net_charge(mol),
                "radicals": mol_radical_count(mol),
                "multiplicity": getattr(mol, "multiplicity", None) or (1 if mol_radical_count(mol) == 0 else 2),
                "is_food": bool(is_food),
                "is_target": False,
                "repr_kind": "adjlist",
                "repr": to_adjlist_mol(mol),
            }
            species_key_to_meta[key] = meta
            if count_as_product:
                if passes_species_filters(mol, heavy_atom_limit, allow_radicals, require_neutral):
                    frontier.append(key)
                else:
                    dropped_counts["product_species_filtered"] += 1
        else:
            if count_as_product:
                if passes_species_filters(mol, heavy_atom_limit, allow_radicals, require_neutral):
                    frontier.append(key)
                else:
                    dropped_counts["product_species_filtered"] += 1
        return key

    r_keys = []
    for sp in rxn.reactants:
        mol = sp.molecule[0] if hasattr(sp, "molecule") and sp.molecule else sp
        r_keys.append(ensure_species_registered(mol, is_food=False, count_as_product=False))

    p_keys = []
    for sp in rxn.products:
        mol = sp.molecule[0] if hasattr(sp, "molecule") and sp.molecule else sp
        p_keys.append(ensure_species_registered(mol, is_food=False, count_as_product=True))

    fam = str(getattr(rxn, "family", "") or "")
    dedup_key = (fam, tuple(sorted(r_keys)), tuple(sorted(p_keys)))
    if dedup_key in rxn_keys:
        dropped_counts["dedup_reaction"] += 1
        return False

    kin_payload = extract_arrhenius_si(rxn, temperature, kin_db=kin_db)
    has_kin = kin_payload is not None
    stats["rxn_generated"] += 1
    if has_kin:
        stats["rxn_with_kinetics"] += 1
    else:
        stats["rxn_without_kinetics"] += 1
        kin_payload = {
            "A": "", "n": "", "Ea_kJmol": "", "T0_K": "",
            "order": len(rxn.reactants),
            "degeneracy": float(getattr(rxn, "degeneracy", 1.0)),
            "units_A": ""
        }

    reactions.append((fam, r_keys, p_keys, kin_payload, has_kin))
    rxn_keys.add(dedup_key)
    return True

def write_reactions_csv(out_path, reactions, species_ids):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "rxn_id", "family", "reactant_ids", "product_ids",
            "order", "degeneracy", "has_kinetics",
            "A", "n", "Ea_kJmol", "T0_K", "units_A", "reversible"
        ])
        for i, (fam, r_keys, p_keys, kin, has_kin) in enumerate(reactions):
            r_ids = [species_ids[k] for k in r_keys]
            p_ids = [species_ids[k] for k in p_keys]
            w.writerow([
                f"R{i}", fam, "|".join(r_ids), "|".join(p_ids),
                int(kin["order"]), float(kin["degeneracy"]),
                "true" if has_kin else "false",
                "" if kin["A"] == "" else f"{float(kin['A']):.6e}",
                "" if kin["n"] == "" else f"{float(kin['n']):.6g}",
                "" if kin["Ea_kJmol"] == "" else f"{float(kin['Ea_kJmol']):.6g}",
                "" if kin["T0_K"] == "" else f"{float(kin['T0_K']):.6g}",
                kin["units_A"] if kin["units_A"] else "",
                "false"
            ])

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    print_generation_config(args)

    thermo_db = ThermoDatabase()
    thermo_db.load(op.join(args.rmg_input_dir, "thermo"), libraries=None, depository=True, surface=False)

    kin_db = KineticsDatabase()
    kin_db.load_families(
        op.join(args.rmg_input_dir, "kinetics", "families"),
        families=args.families,
        depositories=["training"],
    )
    for fam in kin_db.families.values():
        try:
            fam.fill_rules_by_averaging_up(verbose=False)
        except Exception:
            pass

    species_key_to_obj = {}
    species_key_to_meta = {}
    frontier = deque()
    dropped_counts = defaultdict(int)
    stats = defaultdict(int)

    with open(args.food_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            spc = species_from_row(row)
            mol = spc.molecule[0]
            key = to_adjlist_mol(mol)
            if key not in species_key_to_obj:
                species_key_to_obj[key] = spc
                name = row.get("name", "").strip() or to_smiles_mol(mol)
                smiles_pretty = to_smiles_mol(mol)
                charge = mol_net_charge(mol)
                radicals = mol_radical_count(mol)
                multiplicity = getattr(mol, "multiplicity", None)
                species_key_to_meta[key] = {
                    "name": name,
                    "smiles_pretty": smiles_pretty,
                    "charge": charge,
                    "radicals": radicals,
                    "multiplicity": multiplicity if multiplicity is not None else (1 if radicals == 0 else 2),
                    "is_food": True,
                    "is_target": False,
                    "repr_kind": row.get("repr_kind", "smiles"),
                    "repr": row.get("repr", row.get("smiles", "")),
                }
                if passes_species_filters(mol, args.heavy_atom_limit, args.allow_radicals, args.require_neutral):
                    frontier.append(key)
                else:
                    dropped_counts["food_filtered"] += 1

    reactions = []
    rxn_keys = set()
    depth = 0

    while frontier and depth < args.max_depth and len(species_key_to_obj) < args.max_species and len(reactions) < args.max_reactions:
        this_layer = list(frontier)
        frontier.clear()

        for k in this_layer:
            A = species_key_to_obj[k]
            try:
                rxns = kin_db.generate_reactions_from_families(
                    reactants=[A], products=None, only_families=args.families, resonance=True
                )
            except Exception:
                rxns = []
            for rxn in rxns:
                register_reaction(
                    rxn, kin_db, args.temperature,
                    species_key_to_obj, species_key_to_meta, frontier,
                    args.heavy_atom_limit, args.allow_radicals, args.require_neutral,
                    reactions, rxn_keys, stats, dropped_counts
                )
                if len(reactions) >= args.max_reactions or len(species_key_to_obj) >= args.max_species:
                    break
            if len(reactions) >= args.max_reactions or len(species_key_to_obj) >= args.max_species:
                break

        if len(reactions) >= args.max_reactions or len(species_key_to_obj) >= args.max_species:
            break

        pairs_done = 0
        all_current = list(species_key_to_obj.keys())
        for k in this_layer:
            A = species_key_to_obj[k]
            for k2 in all_current:
                B = species_key_to_obj[k2]
                try:
                    rxns = kin_db.generate_reactions_from_families(
                        reactants=[A, B], products=None, only_families=args.families, resonance=True
                    )
                except Exception:
                    rxns = []
                for rxn in rxns:
                    register_reaction(
                        rxn, kin_db, args.temperature,
                        species_key_to_obj, species_key_to_meta, frontier,
                        args.heavy_atom_limit, args.allow_radicals, args.require_neutral,
                        reactions, rxn_keys, stats, dropped_counts
                    )
                    if len(reactions) >= args.max_reactions or len(species_key_to_obj) >= args.max_species:
                        break
                pairs_done += 1
                if pairs_done >= args.pairs_per_depth:
                    dropped_counts["pairs_cap_hit"] += 1
                    break
                if len(reactions) >= args.max_reactions or len(species_key_to_obj) >= args.max_species:
                    break
            if pairs_done >= args.pairs_per_depth:
                break
            if len(reactions) >= args.max_reactions or len(species_key_to_obj) >= args.max_species:
                break

        depth += 1

    species_ids = {}
    species_rows = []
    for idx, key in enumerate(species_key_to_obj.keys()):
        sid = f"S{idx}"
        species_ids[key] = sid
        meta = species_key_to_meta[key]
        species_rows.append([
            sid,
            meta["name"],
            meta["repr_kind"],
            meta["repr"],
            meta["smiles_pretty"],
            meta["charge"],
            meta["radicals"],
            meta["multiplicity"],
            "true" if meta["is_food"] else "false",
            "true" if meta["is_target"] else "false",
        ])

    species_path = op.join(args.out_dir, "species.csv")
    with open(species_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "repr_kind", "repr", "smiles_pretty", "net_charge", "radical_electrons", "multiplicity", "is_food", "is_target"])
        w.writerows(species_rows)

    reactions_path = op.join(args.out_dir, "reactions_kinetics.csv")
    write_reactions_csv(reactions_path, reactions, species_ids)

    print("=== Post-filters and counts ===")
    print(f"Generated reactions total     : {stats['rxn_generated']}")
    print(f"  with kinetics               : {stats['rxn_with_kinetics']}")
    print(f"  without kinetics (kept)     : {stats['rxn_without_kinetics']}")
    print(f"Deduplicated reactions dropped: {dropped_counts['dedup_reaction']}")
    print(f"Food species filtered         : {dropped_counts['food_filtered']}")
    print(f"Product species filtered      : {dropped_counts['product_species_filtered']}")
    print(f"Pairs cap hits                : {dropped_counts['pairs_cap_hit']}")
    print("=== Output ===")
    print(f"Wrote: {species_path}")
    print(f"Wrote: {reactions_path}")
    print(f"Species: {len(species_key_to_obj)}  Reactions: {len(reactions)}")

if __name__ == "__main__":
    main()
