import simulator_modular as sim
import networkx as nx
import matplotlib.pyplot as plt
import pickle

def average(lst):
    return sum(lst) / len(lst)
random_graph_data = []
grid_data = []

# first_run = True
first_run = False
if not first_run:
    # add on top of existing data
    # with open("capture_times_random_graph", "rb") as fp:   # Unpickling
    #     random_graph_data = pickle.load(fp)
    with open("capture_times_grids", "rb") as fp:   # Unpickling
        grid_data = pickle.load(fp)

# generate instances
n = 8
G = nx.grid_2d_graph(n, n)#, periodic=False)
print("TOTAL EDGES BEFORE REMOVING: ", len(G.edges))
for i in range(0, 50, 1):
    if i % 10 == 0: print(i)
    for _ in range(5):
        G = nx.grid_2d_graph(n, n)#, periodic=False)
        sim.add_and_remove_edges(G, i*(1/100), 0)
        max_depth = 200
        capture_times, timeouts = sim.simulate_game(sim.BaseStrategy(G, 1), max_depth)
        print("Timouts skipped:", timeouts)
        grid_data.append((len(G.edges), average(capture_times)))

        G = nx.grid_2d_graph(n, n)#, periodic=False)
        sim.add_and_remove_edges(G, 0, i*(1/100))
        max_depth = 200
        capture_times, timeouts = sim.simulate_game(sim.BaseStrategy(G, 1), max_depth)
        print("Timouts skipped:", timeouts)
        grid_data.append((len(G.edges), average(capture_times)))


# n = 25
# for i in range(0, 100, 1):
#     graph = nx.fast_gnp_random_graph(n, 1- 1/100*i)
#     # i += 0.5
# # for i in range(0, 100+1, 1):
#     # minz_data.append([])
#     # pulp_data.append([])
#     for _ in range(5):
#         graph = nx.fast_gnp_random_graph(n, 1- 1/100*i)
#         edges_plus = [(x+1,y+1) for x,y in graph.edges()]
#         start_fires = [0, 1]
#         # start_fires_plus = [1, 2]
#
#         answers = []
#
#         val, t = sim.simulate.firefighter_minizinc_solve(n, d, graph.edges(), start_fires, maxT)
#         answers.append(int(val))
#         # minz_data[-1].append(t)
#         minz_data.append((i, t))
#         val, t = ffpulp.firefighter_pulp_solve(n, d, edges_plus, start_fires, maxT)
#         answers.append(int(val))
#         # pulp_data[-1].append(t)
#         pulp_data.append((i, t))
#     print(i)


# with open("test", "rb") as fp:   # Unpickling
#     b = pickle.load(fp)

plt.scatter([x for x, y in grid_data], [y for x, y in grid_data])
# plt.scatter(minz_data)
# plt.scatter(pulp_data)
plt.grid()
plt.plot()
plt.waitforbuttonpress()

# with open("capture_times_random_graph", "wb") as fp:   #Pickling
#     pickle.dump(random_graph_data, fp)
with open("capture_times_grids", "wb") as fp:   #Pickling
    pickle.dump(grid_data, fp)
