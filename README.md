# ChemicalExploration

`ChemicalExploration` searches for reaction subnetworks between a given set of food species and desired target species. The search operates under a set of constraints, including F-closure, optional autocatalysis, simple dominance, and basic thermodynamic gating. The core logic is a Mixed-Integer Linear Program (MILP) implemented using PuLP/CBC. Reaction candidates can be generated from SMIRKS templates using RDKit.

## Features

-   **Pathway Discovery**: Identifies viable reaction pathways from food to target compounds.
-   **MILP Core**: Utilizes PuLP with the CBC solver to find constrained, optimal solutions.
-   **Dynamic Reaction Generation**: Leverages RDKit and SMIRKS rules to generate a vast space of possible chemical reactions.
-   **Configurable Constraints**: Applies constraints like F-closure (all reactants of an active reaction must be available), autocatalysis, and simple thermo-gating.
-   **Customizable Environment**: Reaction conditions such as temperature, pH, and ionic strength are fully configurable.

## Project Layout

```
.
+-- proto_meta_search.py      # Main entry point script
+-- real_chem_provider.py     # Provider (loads data, builds network)
¦
+-- config/
¦   +-- env.yaml              # Environment
¦
+-- data/
¦   +-- food.csv              # Input food species
¦   +-- targets.csv           # Desired target species
¦
+-- templates/
¦   +-- smirks_rules.yaml     # SMIRKS reaction rules
¦
+-- external/                 # Optional data (e.g., eQuilibrator dGf' estimates)
¦
+-- output/                   # Output directory for GraphViz .dot files
```

## Setup

This project requires a Conda environment to manage dependencies, particularly RDKit.

1.  **Create Conda Environment**
    Create a new environment named `chem311` with Python 3.11 and RDKit from the `conda-forge` channel.
    ```sh
    conda create -n chem311 -c conda-forge python=3.11 rdkit
    ```

2.  **Activate Environment**
    ```sh
    conda activate chem311
    ```

3.  **Install Python Dependencies**
    Install the remaining required packages using pip.
    ```sh
    pip install numpy pandas pyyaml pulp
    ```

## Usage

To start a search, activate the Conda environment and run the main script.

```sh
conda activate chem311
python proto_meta_search.py
```

If a valid subnetwork is found, GraphViz `.dot` files will be written to the `output/` directory.

## Configuration

Before running, configure the input files to define your search space and conditions.

### Food Species (`data/food.csv`)

List the available starting molecules in this file.
* **Format**: CSV with columns: `id`, `name`, `smiles`.
* **Note**: Water (`O`) is added internally by default. Only list it here if you want it to be treated as a buffered resource.

*Example: `data/food.csv`*
```csv
id,name,smiles
ac,acetate,CC(=O)[O-]
glc,glucose,C([C@@H]1[C@H]([C@@H]([C@H]([C@H](O1)O)O)O)O)O
```

### Target Species (`data/targets.csv`)

List the desired product molecules in this file.
* **Format**: CSV with columns: `id`, `name`, `smiles`.
* **Note**: By default, the first target listed in this file is used as the autocatalyst (X).

*Example: `data/targets.csv`*
```csv
id,name,smiles
mal,malate,C(C(C(=O)[O-])O)C(=O)[O-]
```

### Environment (`config/env.yaml`)

This file controls the simulation environment parameters

*Example: `config/env.yaml`*
```yaml
temperature_K: 298.15
pH: 7.0
ionic_strength_M: 0.1
food_budget: 50.0
smirks_rules: templates/smirks_rules.yaml
```

### SMIRKS Rules (`templates/smirks_rules.yaml`)

Define the chemical transformations that RDKit will use to generate reactions.
* **Format**: A YAML list of rule objects.
* **Fields per rule**:
    * `id`: A unique identifier for the rule.
    * `reversible`: `true` or `false`.
    * `smirks`: The SMIRKS string describing the reaction.
    * `max_pairs` (optional): Caps the number of reactant pairs considered for this rule to limit expansion.
    * `max_outcomes_per_pair` (optional): Caps the number of products generated from a single reactant pair.

*Example: `templates/smirks_rules.yaml`*
```yaml
- id: amide_bond_formation
  reversible: false
  smirks: "[C:1](=O)[S:2][C:3].[N:4]>>[C:1](=O)[N:4].[S:2][C:3]"
  max_pairs: 64
  max_outcomes_per_pair: 4
```
**Note**: The RDKit engine currently supports rules with 1–2 reactant patterns.

## Notes & Troubleshooting

* **File Encoding**: All CSV and YAML files **must be UTF-8 encoded**.
* **Python Version**: RDKit wheels via pip are not generally available for Python 3.12. Using the specified Conda environment with Python 3.11 is strongly recommended.
* **Quick RDKit Check**: To verify your RDKit installation is working correctly within the environment, run:
    ```sh
    conda activate chem311
    python -c "from rdkit import rdBase; print(rdBase.rdkitVersion)"
    ```