# rmg_test.py -- RMG-Py 3.3.0 smoke test (ASCII only)
import os
import os.path as op
from rmgpy.data.thermo import ThermoDatabase
from rmgpy.data.kinetics import KineticsDatabase
from rmgpy.species import Species
from rmgpy.molecule import translator as tr  # canonical SMILES I/O

RMG_INPUT = os.environ.get("RMG_INPUT_DIR", "/rmgdb/input")
print("RMG input dir:", RMG_INPUT)

# Load thermo (safe)
thermo_db = ThermoDatabase()
thermo_db.load(op.join(RMG_INPUT, "thermo"), libraries=None, depository=True, surface=False)
print("Thermo loaded.")

# Load one small kinetics family for speed
kin_db = KineticsDatabase()
families_dir = op.join(RMG_INPUT, "kinetics", "families")
kin_db.load_families(families_dir, families=["Ketoenol"], depositories=["training"])
print("Loaded families:", list(kin_db.families.keys()))

# Build a Species from SMILES (acetone)
spc = Species()
spc.from_smiles("CC(=O)C")

# Generate reactions via KineticsDatabase (new API)
rxns = kin_db.generate_reactions_from_families(
    reactants=[spc],
    products=None,
    only_families=["Ketoenol"],
    resonance=True
)
print("Generated %d reaction(s) via KineticsDatabase." % len(rxns))

def to_smiles_any(x):
    # Accept Species or Molecule and return canonical SMILES via translator
    if hasattr(x, "molecule") and x.molecule:
        return tr.to_smiles(x.molecule[0])
    return tr.to_smiles(x)

for i, rxn in enumerate(rxns[:5]):
    r = " + ".join(to_smiles_any(m) for m in rxn.reactants)
    p = " + ".join(to_smiles_any(m) for m in rxn.products)
    print("r%d: %s -> %s" % (i, r, p))

print("RMG smoke test: OK")
