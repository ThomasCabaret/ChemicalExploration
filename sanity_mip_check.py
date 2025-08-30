# sanity_mip_check.py
# Minimal sanity check for PuLP + CBC on Windows.

import sys, os
print("Python:", sys.version.replace("\n", " "))

try:
    import pulp
    from pulp import LpProblem, LpMaximize, LpVariable, LpBinary, LpContinuous, lpSum, PULP_CBC_CMD, LpStatus, value
    print("PuLP version:", getattr(pulp, "__version__", "unknown"))
except Exception as e:
    print("ImportError:", e)
    print('Fix:  python -m pip install --upgrade pip && python -m pip install pulp')
    raise

# Try to expose PuLP-bundled cbc.exe to PATH (harmless if already there)
try:
    base = os.path.dirname(pulp.__file__)
    for sub in (("solverdir","cbc","win","64"), ("solverdir","cbc","win","32")):
        cbc_dir = os.path.join(base, *sub)
        if os.path.isdir(cbc_dir):
            os.environ["PATH"] = cbc_dir + os.pathsep + os.environ.get("PATH","")
            print("CBC bin dir added to PATH:", cbc_dir)
            break
except Exception:
    pass

# Tiny MILP: maximize x + y s.t. x + 2y <= 2, x,y binary
model = LpProblem("sanity", LpMaximize)
x = LpVariable("x", lowBound=0, upBound=1, cat=LpBinary)
y = LpVariable("y", lowBound=0, upBound=1, cat=LpBinary)
model += x + y
model += x + 2*y <= 2

solver = PULP_CBC_CMD(msg=1, timeLimit=30)
status_code = model.solve(solver)
status_str = LpStatus.get(status_code, str(status_code))

print("Status:", status_str)
if status_str == "Optimal":
    print("Objective:", value(model.objective))
    print("x =", int(round(value(x))), "y =", int(round(value(y))))
else:
    print("Solver did not find Optimal. Check CBC installation or PATH.")
