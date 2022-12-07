from interaction_model import *
import networkx as nx

def make_new_nodes(G: nx.Graph, n=5):
    first_new_node = G.number_of_nodes()+1
    new_nodes = [first_new_node+i for i in range(n)]
    G.add_nodes_from(new_nodes)
    return new_nodes

def create_clique(G: nx.Graph, n=5):
    new_nodes = make_new_nodes(G, n)
    G.add_edges_from([(i, j) for i in new_nodes for j in new_nodes if i != j])
    return new_nodes

def create_path(G: nx.Graph, n=5, loop=False):
    new_nodes = make_new_nodes(G, n)
    G.add_edges_from([(new_nodes[i], new_nodes[i-1]) for i in range(1, len(new_nodes))])
    if loop:
        G.add_edge(new_nodes[0], new_nodes[-1])
    return new_nodes

def create_grid_square(G: nx.Graph, n=5, m=5, diagonals=0, skip_ngon=1):
    prev_new_nodes = create_path(G, n)
    all_new_nodes = prev_new_nodes
    for i in range(1,m):
        new_nodes = create_path(G, n)
        G.add_edges_from([(prev_new_nodes[i], new_nodes[i]) for i in range(0,n, skip_ngon)])
        if diagonals >= 1:
            G.add_edges_from([(prev_new_nodes[i-1], new_nodes[i]) for i in range(1, n)])
        if diagonals >= 2:
            G.add_edges_from([(prev_new_nodes[i+1], new_nodes[i]) for i in range(n-1)])
        all_new_nodes.extend(new_nodes)
        prev_new_nodes = new_nodes
    return all_new_nodes

def create_grid_hex(G: nx.Graph, n=5, m=5):
    n = 2*n+1
    m = 2*m

    prev_new_nodes = create_path(G, n)
    all_new_nodes = prev_new_nodes
    for i in range(1,m):
        new_nodes = create_path(G, n)
        G.add_edges_from([(prev_new_nodes[i], new_nodes[i]) for i in range((i-1)%2,n, 2)])
        all_new_nodes.extend(new_nodes)
        prev_new_nodes = new_nodes
    return all_new_nodes

    # return create_grid_square(G, 2*n, 1+2*m, skip_ngon=2)

def create_sample(G: nx.Graph):
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
    G.add_edges_from([(1,2), (3,4), (2,5), (4,5), (6,7), (8,9), (4,7), (1,7), (3,5), (2,7), (5,8), (2,9), (5,7)])


def mark_corner(G: nx.Graph, node: int):
    G.nodes[node]["title"] = "corner"
    G.nodes[node]["group"] = 2
    print("corner marked:", node)

def is_corner(G: nx.Graph, node: int) -> bool:
    return G.nodes[node].get("title", "") == "corner"

def find_corners(G: nx.Graph):
    candidates = set(G.nodes())
    # print(candidates)
    while candidates:
        corner_node = candidates.pop()
        if not G[corner_node]:
            mark_corner(G, corner_node)
            continue
        corner_coverage = set([corner_node] + [x for x in G[corner_node] if not is_corner(G, x)])
        is_this_corner = False
        for other_node in G[corner_node]:
            matches = 1
            for back_track_node in G[other_node]:
                if back_track_node == corner_node or back_track_node in corner_coverage:
                    matches += 1
            if matches == len(corner_coverage):
                is_this_corner = True
                break
        if is_this_corner:
            mark_corner(G, corner_node)
            for node in corner_coverage:
                if not is_corner(G, node) and node not in candidates:
                    candidates.add(node)
    print(G.nodes[5])


