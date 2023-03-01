# from pyvis.network import Network
import networkx as nx

def simulate(G):
    escape_G = G.copy()
    # pick cop position
    cop_pos = 0
    results = []

    #TODO: find "ideal" robber position - largest depth, for comparison of results


    # pick robber position
    for robber_pos in range(len(G)):
        if robber_pos == cop_pos: continue
        # simulate gameplay
        result = simulate_nonconsec(G, escape_G, cop_pos, robber_pos)
        results.append(result["worst_capture_time"])
        print(" ;", results[-1], end="")
    result_a = "No consecutive roadblocks result:", sum(results) / len(G)
    escape_G = G.copy()

    print("### next ###")
    results = []
    roadblocks_max = 1
    for robber_pos in range(len(G)):
        if robber_pos == cop_pos: continue
        result = simulate_nonconsec(G, escape_G, cop_pos, robber_pos, roadblocks_max)
        results.append(result["worst_capture_time"])
        print(" ;", results[-1], end="")
    result_b = f"No consecutive upto { roadblocks_max } roadblocks result:", sum(results) / len(G)
    escape_G = G.copy()

    print("### next ###")

    results = []
    for robber_pos in range(len(G)):
        if robber_pos == cop_pos: continue
        result = simulate_nonconsec(G, escape_G, cop_pos, robber_pos, 0)
        results.append(result["worst_capture_time"])
        print(" ;", results[-1], end="")
    print()
    # result_c = "No consecutive roadblocks result:", sum(results) / len(G)

    print(*result_a)
    print(*result_b)
    print("No roadblocks: ", sum(results) / len(G))


def simulate_nonconsec(G, escape_G, cop_pos=0, robber_pos=0, blocks=-1,
        removed_edges = []):
    # print("blocks = ", blocks)
    escape_G = escape_G.copy()
    removed_edges = removed_edges.copy()
    worst_capture_time = 0
    capture_time = 0
    # print(cop_pos, robber_pos)
    while robber_pos != cop_pos:
        capture_time += 1

        # cop moves
        cop_pos = nx.shortest_path(G, cop_pos, robber_pos)[1]
        if robber_pos == cop_pos:
            break

        # cop blocks edges
        dist = len(nx.shortest_path(G, cop_pos, robber_pos))
        new_removed_edges = []
        neighbors = nx.neighbors(G, robber_pos)
        for v in neighbors:
            if len(new_removed_edges) >= blocks and blocks != -1: break
            if len(nx.shortest_path(G, cop_pos, v)) > dist and escape_G.has_edge(robber_pos, v):
                escape_G.remove_edge(robber_pos, v)
                # print("remove", (robber_pos, v), end="; ")
                new_removed_edges.append((robber_pos, v))

        # cop unblocks old roadblocks
        for v, w in removed_edges:
            # print("add", (v, w), end="; ")
            escape_G.add_edge(v, w)
        removed_edges = new_removed_edges

        # robber moves
        capture_times = [0]
        neighbors = list(nx.neighbors(escape_G, robber_pos))
        for v in neighbors:
            if len(nx.shortest_path(G, cop_pos, v)) > dist:
                capture_times.append(simulate_nonconsec(G, escape_G, cop_pos, v, blocks, list(removed_edges))["worst_capture_time"])

        worst_capture_time = max(worst_capture_time, capture_time + max(capture_times))

    worst_capture_time = max(worst_capture_time, capture_time)
    # print("DONE? .. ", worst_capture_time)
    return {
        "worst_capture_time": worst_capture_time
    }

n = 13
print(f"random_tree({ n })")
simulate(nx.random_tree(n))

n = 30
print(f"random_tree({ n })")
simulate(nx.random_tree(n))

n = 13
print(f"path_graph({ n })")
simulate(nx.path_graph(n))
