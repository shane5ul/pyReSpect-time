
from pyrespect_time import ReSpect, ReSpectConfig

# Default settings — fit from a data file
solver = ReSpect()
solver.fit("Gt.dat")  # "Gt.dat" file contains data

# Access results
print(solver.continuous.H)    # continuous spectrum H(s)
print(solver.discrete.tau)    # discrete relaxation times
print(solver.discrete.g)      # discrete mode weights

# Save and plot
solver.save(which="base", path="output/")
solver.plot(which="base", toFile=True, path="output/")
