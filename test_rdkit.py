# test_rdkit.py
from rdkit import Chem
import rdkit
from rdkit.Chem import rdChemReactions as Reactions

def ok(b, msg):
    print(f"[OK] {msg}" if b else f"[FAIL] {msg}")

print(f"RDKit version: {rdkit.__version__}")

# 1) Sanity: SMILES round-trip
ethanol = Chem.MolFromSmiles("CCO")
ok(ethanol is not None, "MolFromSmiles('CCO')")
print("Canonical CCO ->", Chem.MolToSmiles(ethanol, canonical=True))

# 2) Tiny reaction test (amide formation template, purely formal)
#    Acetic acid + ammonia -> acetamide + water (as a template)
smirks = "[C:1](=O)O.[N:2]>>[C:1](=O)N.[O:3][H]"
rxn = Reactions.ReactionFromSmarts(smirks)
ok(rxn is not None, "ReactionFromSmarts parsed")

reactants = (Chem.MolFromSmiles("CC(=O)O"), Chem.MolFromSmiles("N"))
ok(all(r is not None for r in reactants), "Reactant SMILES parsed")

products_sets = rxn.RunReactants(reactants)
ok(len(products_sets) >= 1, "Reaction produced at least one product set")

# Print the first product set as SMILES
if products_sets:
    prods = [Chem.MolToSmiles(p, canonical=True) for p in products_sets[0]]
    print("Products (first set):", prods)

print("RDKit basic test completed.")
