import matplotlib.pyplot as plt
import pickle

with open("capture_times_grids", "rb") as fp:   # Unpickling
    grid_data = pickle.load(fp)
# with open("benchmark_data_pulp", "rb") as fp:   # Unpickling
#     pulp_data = pickle.load(fp)


# plt.scatter([1-x/100 for x, y in minz_data], [y for x, y in minz_data], label="minizinc", s=1.5)
# plt.scatter([1-x/100 for x, y in pulp_data], [y for x, y in pulp_data], label="PuLP", s=1.5)
plt.scatter([x for x, y in grid_data], [y for x, y in grid_data], label="Grid graph", s=5)
# plt.scatter(minz_data)
# plt.scatter(pulp_data)
plt.axvline(112)
plt.grid()
plt.legend()
plt.ylabel("Capture time (cop turns), timeout=200")
plt.xlabel("edge count (left: grid with only removed edges, right: grid with only added edges)")
plt.title("Capture time of (1r, 1c, 1b)-instances on an 8x8 grid; \naverage of greedy strategy runs where cop starts at (0,0) \nand robber tries every other node as starting position")
# plt.scatter(minz_data)
# plt.scatter(pulp_data)
plt.plot()
plt.waitforbuttonpress()

plt.savefig("grid_sim.png", dpi=150)
