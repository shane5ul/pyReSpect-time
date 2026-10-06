"""Minimal example. Run from the repository root:  python quickstart.py"""
from pyrespect_time import ReSpect

# Default settings; fit from a data file (arrays work too: solver.fit(t, Gt))
solver = ReSpect()
solver.fit("tests/test2.dat")

# Access results
print(solver.continuous.H)    # log of the continuous spectrum, H(s)
print(solver.discrete.tau)    # discrete relaxation times
print(solver.discrete.g)      # discrete mode weights

# Save and plot
solver.save(which="full", path="output/")
figs = solver.plot(which="full", toFile=True, path="output/")
