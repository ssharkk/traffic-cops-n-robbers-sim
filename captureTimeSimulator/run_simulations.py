import simulator_modular as sim
import networkx as nx
import matplotlib.pyplot as plt
import time
from timeit import default_timer as timer
import pickle
import pandas as pd
import numpy as np
from tulip import tlp

def average(lst):
    if len(lst) == 0: return 0
    return sum(lst) / len(lst)
random_graph_data = []
grid_data = []

CODENAMES = ["stoch", "inconv", "tt_planar", "planar", "accumul", "optimal"]
MAX_DEPTH = 200

first_run = False
# first_run = True

# if not first_run:
#     # add on top of existing data
#     # with open("capture_times_random_graph", "rb") as fp:   # Unpickling
#     #     random_graph_data = pickle.load(fp)
#     with open("capture_times_grids", "rb") as fp:   # Unpickling
#         grid_data = pickle.load(fp)

def save_append_data(filename, edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken):
    global first_run

    # data = {
    #     "edges": edges_size,
    #     "ct_stoch": capture_times["stoch"],
    #     "ct_inconv": capture_times["inconv"],
    #     "ct_planar": capture_times["planar"],
    #     "ct_accumul": capture_times["accumul"],
    #     "ct_optimal": capture_times["optimal"],
    #     "to_stoch": timeouts["stoch"],
    #     "to_inconv": timeouts["inconv"],
    #     "to_planar": timeouts["planar"],
    #     "to_accumul": timeouts["accumul"],
    #     "to_optimal": timeouts["optimal"],
    #     "bu_stoch": roadblock_uses["stoch"],
    #     "bu_inconv": roadblock_uses["inconv"],
    #     "bu_planar": roadblock_uses["planar"],
    #     "bu_accumul": roadblock_uses["accumul"],
    #     "bu_optimal": roadblock_uses["optimal"],
    #     "bi_stoch": block_inconvenience["stoch"],
    #     "bi_inconv": block_inconvenience["inconv"],
    #     "bi_planar": block_inconvenience["planar"],
    #     "bi_accumul": block_inconvenience["accumul"],
    #     "bi_optimal": block_inconvenience["optimal"],
    #     "tt_stoch": time_taken["stoch"],
    #     "tt_inconv": time_taken["inconv"],
    #     "tt_planar": time_taken["planar"],
    #     "tt_accumul": time_taken["accumul"],
    #     "tt_optimal": time_taken["optimal"],
    # }
    data = {
        "edges": np.array(edges_size*5),
        "ct": np.concatenate((capture_times["stoch"],
            capture_times["inconv"],
            capture_times["planar"],
            capture_times["accumul"],
            capture_times["optimal"])),
        "to": np.concatenate((timeouts["stoch"],
            timeouts["inconv"],
            timeouts["planar"],
            timeouts["accumul"],
            timeouts["optimal"])),
        "bu": np.concatenate((roadblock_uses["stoch"],
            roadblock_uses["inconv"],
            roadblock_uses["planar"],
            roadblock_uses["accumul"],
            roadblock_uses["optimal"])),
        "bi": np.concatenate((block_inconvenience["stoch"],
            block_inconvenience["inconv"],
            block_inconvenience["planar"],
            block_inconvenience["accumul"],
            block_inconvenience["optimal"])),
        "tt": np.concatenate((time_taken["stoch"],
            time_taken["inconv"],
            time_taken["planar"],
            time_taken["accumul"],
            time_taken["optimal"])),
        "strat": np.array(
            ["stochastic"] * len(capture_times["stoch"])
            +["min inconvenience"] * len(capture_times["inconv"])
            +["planar heuristic"] * len(capture_times["planar"])
            +["saving"] * len(capture_times["accumul"])
            +["optimal unlimited"] * len(capture_times["optimal"])
        )
    }
    print(data)
    print(data["edges"].shape)
    print(data["ct"].shape)
    print(data["to"].shape)
    print(data["bu"].shape)
    print(data["bi"].shape)
    print(data["tt"].shape)
    print(data["strat"].shape)
    df = pd.DataFrame(data)
    df.to_csv(filename, mode='a', index=False, header=first_run)

def run_strategy(strategy, code_name, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken):
    time_a = timer()
    ct, to, bu, bi = sim.simulate_game(strategy, MAX_DEPTH)
    time_b = timer()
    tt = time_b - time_a
    capture_times[code_name].append(average(ct))
    timeouts[code_name].append(to)
    roadblock_uses[code_name].append(average(bu))
    block_inconvenience[code_name].append(average(bi))
    time_taken[code_name].append(tt)

def run_all_strategies(G, layout, edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken):
    edges_size.append(len(G.edges))
    roadblocks_limit = 3
    run_strategy(sim.GreedyStochasticStrategy(G, roadblocks_limit, layout), "stoch", capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    run_strategy(sim.GreedyMinInconvenienceStrategy(G, roadblocks_limit, layout), "inconv", capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    run_strategy(sim.GreedyPlanarStrategy(G, roadblocks_limit, layout), "planar", capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    run_strategy(sim.GreedySavingStrategy(G, roadblocks_limit, layout), "accumul", capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    run_strategy(sim.OptimalUnlimitedRoadblocksStrategy(G, layout), "optimal", capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)

def run_grid_simulations():
    capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken = {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}
    edges_size = []
    n = 8
    print("NEXT: run_grid_simulations")

    G = nx.grid_2d_graph(n, n)#, periodic=False)
    layout = nx.planar_layout(G)
    run_all_strategies(G, layout, edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    for i in range(0, 50, 1):
        if i % 10 == 0: print(i)
        for _ in range(30):
        # for _ in range(1):
            G = nx.grid_2d_graph(n, n)#, periodic=False)
            sim.add_and_remove_edges(G, i*(1/100), 0)
            run_all_strategies(G, layout, edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    for i in range(0, 50, 1):
        if i % 10 == 0: print(i)
        for _ in range(30):
        # for _ in range(1):
            G = nx.grid_2d_graph(n, n)#, periodic=False)
            sim.add_and_remove_edges(G, 0, i*(1/100))
            run_all_strategies(G, layout, edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    save_append_data("grid_data3.csv", edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)

def run_gnm_simulations():
    capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken = {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}
    edges_size = []
    n = 64
    print("NEXT: run_gnm_simulations")

    # for i in range(3, n, 1):
    for i in range(1, 100, 1):
        for _ in range(30):
            G = nx.gnm_random_graph(n, 80+i)#, periodic=False)
            # G = nx.connected_watts_strogatz_graph(n, 4, i/100)#, periodic=False)
            layout = nx.fruchterman_reingold_layout(G, seed=1, iterations=1500)
            run_all_strategies(G, layout, edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
        # if i % 10 == 0: print(i)
        if i % 4 == 0: print(i)
        # print(i)
    # save_append_data("watts_strogatz_data.csv", edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    save_append_data("gnm_data3.csv", edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)

def run_newman_watts_strogatz_simulations():
    capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken = {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}
    edges_size = []
    n = 64
    print("NEXT: run_newman_watts_strogatz_simulations")
    # for i in range(3, n, 1):
    for j in range(2, 6):
        for i in range(1, 100, 1):
            for _ in range(3):
                # G = nx.gnm_random_graph(n, 80+i)#, periodic=False)
                # G = nx.connected_watts_strogatz_graph(n, j, i/100, 100)#, periodic=False)
                G = nx.newman_watts_strogatz_graph(n, j, i/100)#, periodic=False)
                # break
                layout = nx.fruchterman_reingold_layout(G, seed=1, iterations=1500)
                run_all_strategies(G, layout, edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
        # if i % 10 == 0: print(i)
            if i % 4 == 0: print(i)
        # print(i)
    # save_append_data("watts_strogatz_data.csv", edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    save_append_data("watts_strogatz_data3.csv", edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)

def run_planar_simulations():
    capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken = {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}, {c:[] for c in CODENAMES}
    edges_size = []
    params = tlp.getDefaultPluginParameters('Planar Graph')
    n = 64
    print("NEXT: run_planar_simulations")

    # e = 186
    params["nodes"] = n
    # for i in range(3, n, 1):
    for i in range(0, 122, 1):
        # create new planar graph ( max edge density )
        for _ in range(30):
            tlp_graph = tlp.importGraph('Planar Graph', params)
            G = nx.Graph()
            # convert to networkx graph
            for n in tlp_graph.getNodes():
                G.add_node(n.id)
            for e in tlp_graph.getEdges():
                G.add_edge(tlp_graph.source(e).id, tlp_graph.target(e).id)
            sim.remove_n_edges_connected(G, i)
            # print(len(G.nodes), len(G.edges), len(nx.difference(G, prevG).edges), nx.is_isomorphic(prevG, G))#, nx.graph_edit_distance(prevG, G, timeout=5))
            # G = nx.connected_watts_strogatz_graph(n, 4, i/100)#, periodic=False)
            layout = nx.planar_layout(G)
            run_all_strategies(G, layout, edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
        # if i % 10 == 0: print(i)
        if i % 4 == 0: print(i)
        # print(i)
    # save_append_data("watts_strogatz_data.csv", edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)
    save_append_data("planar_data3.csv", edges_size, capture_times, timeouts, roadblock_uses, block_inconvenience, time_taken)



if __name__ == "__main__":
    # run_grid_simulations()
    run_gnm_simulations()
    run_newman_watts_strogatz_simulations()
    run_planar_simulations()

# generate instances
# n = 8
# G = nx.grid_2d_graph(n, n)#, periodic=False)
# print("TOTAL EDGES BEFORE REMOVING: ", len(G.edges))
# for i in range(0, 50, 1):
#     if i % 10 == 0: print(i)
#     for _ in range(5):
#         # G = nx.grid_2d_graph(n, n)#, periodic=False)
#         # sim.add_and_remove_edges(G, i*(1/100), 0)
#         # MAX_DEPTH = 200
#         # capture_times, timeouts = sim.simulate_game(sim.BaseStrategy(G, 1), MAX_DEPTH)
#         # print("Timouts skipped:", timeouts)
#         # grid_data.append((len(G.edges), average(capture_times)))
#         #
#         # G = nx.grid_2d_graph(n, n)#, periodic=False)
#         # sim.add_and_remove_edges(G, 0, i*(1/100))
#         # MAX_DEPTH = 200
#         # capture_times, timeouts = sim.simulate_game(sim.BaseStrategy(G, 1), MAX_DEPTH)
#         # print("Timouts skipped:", timeouts)
#         # grid_data.append((len(G.edges), average(capture_times)))
#
#
#         G = nx.grid_2d_graph(n, n)#, periodic=False)
#         sim.add_and_remove_edges(G, i*(1/100), 0)
#         MAX_DEPTH = 200
#         capture_times, timeouts = sim.simulate_game(sim.GreedySavingStrategy(G, 1), MAX_DEPTH)
#         print("Timouts skipped:", timeouts)
#         grid_data.append((len(G.edges), average(capture_times)))
#         #
#         G = nx.grid_2d_graph(n, n)#, periodic=False)
#         sim.add_and_remove_edges(G, 0, i*(1/100))
#         MAX_DEPTH = 200
#         capture_times, timeouts = sim.simulate_game(sim.GreedySavingStrategy(G, 1), MAX_DEPTH)
#         print("Timouts skipped:", timeouts)
#         grid_data.append((len(G.edges), average(capture_times)))
#
#     G = nx.grid_2d_graph(n, n)#, periodic=False)
#     sim.add_and_remove_edges(G, i*(1/100), 0)
#     MAX_DEPTH = 200
#     capture_times, timeouts = sim.simulate_game(sim.OptimalUnlimitedRoadblocksStrategy(G), MAX_DEPTH)
#     print("Timouts skipped:", timeouts)
#     grid_data.append((len(G.edges), average(capture_times)))
#
#     G = nx.grid_2d_graph(n, n)#, periodic=False)
#     sim.add_and_remove_edges(G, 0, i*(1/100))
#     MAX_DEPTH = 200
#     capture_times, timeouts = sim.simulate_game(sim.OptimalUnlimitedRoadblocksStrategy(G), MAX_DEPTH)
#     print("Timouts skipped:", timeouts)
#     grid_data.append((len(G.edges), average(capture_times)))
#
# def compile_output(data, G, capture_times, timouts, time_taken, ):
#     capture_times = []
#
# # n = 25
# # for i in range(0, 100, 1):
# #     graph = nx.fast_gnp_random_graph(n, 1- 1/100*i)
# #     # i += 0.5
# # # for i in range(0, 100+1, 1):
# #     # minz_data.append([])
# #     # pulp_data.append([])
# #     for _ in range(5):
# #         graph = nx.fast_gnp_random_graph(n, 1- 1/100*i)
# #         edges_plus = [(x+1,y+1) for x,y in graph.edges()]
# #         start_fires = [0, 1]
# #         # start_fires_plus = [1, 2]
# #
# #         answers = []
# #
# #         val, t = sim.simulate.firefighter_minizinc_solve(n, d, graph.edges(), start_fires, maxT)
# #         answers.append(int(val))
# #         # minz_data[-1].append(t)
# #         minz_data.append((i, t))
# #         val, t = ffpulp.firefighter_pulp_solve(n, d, edges_plus, start_fires, maxT)
# #         answers.append(int(val))
# #         # pulp_data[-1].append(t)
# #         pulp_data.append((i, t))
# #     print(i)
#
#
# # with open("test", "rb") as fp:   # Unpickling
# #     b = pickle.load(fp)
#
# plt.scatter([x for x, y in grid_data], [y for x, y in grid_data])
# # plt.scatter(minz_data)
# # plt.scatter(pulp_data)
# plt.grid()
# plt.plot()
# plt.waitforbuttonpress()
#
# # with open("capture_times_random_graph", "wb") as fp:   #Pickling
# #     pickle.dump(random_graph_data, fp)
# with open("capture_times_grids", "wb") as fp:   #Pickling
#     pickle.dump(grid_data, fp)
