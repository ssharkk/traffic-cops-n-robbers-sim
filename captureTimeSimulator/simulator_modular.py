# from pyvis.network import Network
import networkx as nx
from collections import namedtuple, Counter
from pprint import pprint
import random

GameState = namedtuple("GameState", "turn c_pos r_pos blocked_edges prev_state")

class BaseStrategy():
    def __init__(self, G, roadblocks = 0):
        self.roadblocks = roadblocks
        self.source_graph = G.copy()
        # self.edge_index = G.edges

    def get_initial_state(self):
        assert len(self.source_graph) > 0, "Empty source graph"
        return [GameState("init", None, None, (), None)]

    def get_cop_start_position(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        new_c_pos = list(self.source_graph.nodes)[len(self.source_graph)//2] # mid node
        new_c_pos = list(self.source_graph.nodes)[0] # start node
        return [GameState("place_c", new_c_pos, r_pos, blocked_edges, state)]

    def get_robber_start_position(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        return [GameState("place_r", c_pos, new_r_pos, blocked_edges, state) for new_r_pos in self.source_graph.nodes if new_r_pos != c_pos]

    def play_cop(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        cur_dist = len(nx.shortest_path(self.source_graph, c_pos, r_pos))
        moves = [new_c_pos for new_c_pos in nx.neighbors(self.source_graph, c_pos) if len(nx.shortest_path(self.source_graph, new_c_pos, r_pos)) < cur_dist]
        plays = [GameState("play_c", new_c_pos, r_pos, blocked_edges, state) for new_c_pos in moves]
        # print("cop", c_pos, moves)
        return [random.choice(plays)]#[:1]

    def play_robber(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        cur_dist = len(nx.shortest_path(self.source_graph, c_pos, r_pos))
        for a, b in blocked_edges:
            # print("remove", a, b, "len", len(self.source_graph.edges))
            self.source_graph.remove_edge(a, b)
        moves = [new_r_pos for new_r_pos in nx.neighbors(self.source_graph, r_pos)]
        for a, b in blocked_edges:
            self.source_graph.add_edge(a, b)
            # print("add", a, b, "len", len(self.source_graph.edges))
        moves = [(len(nx.shortest_path(self.source_graph, c_pos, new_r_pos)), new_r_pos) for new_r_pos in moves]
        best_loss = max(moves)[0]
        moves = [move[1] for move in moves if move[0] == best_loss]
        plays = [GameState("play_r", c_pos, new_r_pos, blocked_edges, state) for new_r_pos in moves]
        # print("robber", r_pos, moves, list(nx.neighbors(self.source_graph, r_pos)))
        return [random.choice(plays)] #[:1]

    def play_roadblocks(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        cur_dist = len(nx.shortest_path(self.source_graph, c_pos, r_pos))
        new_blocks = [(r_pos, v) for v in nx.neighbors(self.source_graph, r_pos) if len(nx.shortest_path(self.source_graph, c_pos, v)) > cur_dist]
        # print("roadblock", len(new_blocks))
        new_blocks = tuple(new_blocks[:self.roadblocks])
        return [GameState("play_b", c_pos, r_pos, new_blocks, state)]

    def eval_winner(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        return c_pos == r_pos

    def capture_time(self, state):
        time = 0
        c, r = None, None
        while state.prev_state is not None:
            if state.turn == "play_c":
                time += 1
                c, r = state.prev_state.c_pos, state.prev_state.r_pos
            state = state.prev_state
        return time, c, r

def simulate_game(G_strategy, max_depth):
    ends = 0
    # global sim_settings
    cur_states = G_strategy.get_initial_state()
    results = {}
    best_trace = []
    all_traces = []
    trace_capturetimes = []

    while len(cur_states) > 0:
        state = cur_states.pop()
        if state.turn == "init":
            cur_states.extend(G_strategy.get_cop_start_position(state))
        elif state.turn == "place_c":
            cur_states.extend(G_strategy.get_robber_start_position(state))
        elif state.turn in ["play_r","place_r"]:
            cur_states.extend(G_strategy.play_cop(state))
        elif state.turn in ["play_c","place_c"]:
            cur_states.extend(G_strategy.play_roadblocks(state))
        elif state.turn in ["play_b"]:
            cur_states.extend(G_strategy.play_robber(state))

        # print(G_strategy.capture_time(state), state.c_pos, state.r_pos)
        while len(cur_states) > 0 and (G_strategy.eval_winner(cur_states[-1]) or G_strategy.capture_time(cur_states[-1])[0] >= max_depth):
            win_state = cur_states.pop()
            all_traces.append(win_state)
            trace_capturetimes.append(G_strategy.capture_time(win_state)[0])
            # print(trace_capturetimes[-1])
    print(len(all_traces), "WAYS FOR COP TO WIN! Average turns:", sum(trace_capturetimes) / len(all_traces))
    capture_times_counter = Counter(trace_capturetimes)
    # pprint(capture_times_counter)
    return [x for x in trace_capturetimes if x < max_depth], capture_times_counter.get(max_depth, 0)


class GridStrategy(BaseStrategy):
    def play_cop(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        cur_dist = len(nx.shortest_path(self.source_graph, c_pos, r_pos))
        moves = [new_c_pos for new_c_pos in nx.neighbors(self.source_graph, c_pos) if len(nx.shortest_path(self.source_graph, new_c_pos, r_pos)) < cur_dist]
        if c_pos[0] < r_pos[0]:
            moves = [(c_pos[0]+1, c_pos[1])]
        plays = [GameState("play_c", new_c_pos, r_pos, blocked_edges, state) for new_c_pos in moves]
        # print("cop", c_pos, moves)
        return plays[:1]

    def play_roadblocks(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        cur_dist = len(nx.shortest_path(self.source_graph, c_pos, r_pos))
        new_blocks = [(r_pos, v) for v in nx.neighbors(self.source_graph, r_pos) if len(nx.shortest_path(self.source_graph, c_pos, v)) > cur_dist]
        if c_pos[0]+1 >= r_pos[0] and r_pos[0] > 0:
            new_blocks = [((r_pos[0]-1, r_pos[1]), (r_pos[0], r_pos[1]))]
            # print(new_blocks)
            # print("c,r", c_pos, r_pos)
        elif r_pos[0] == 0:
            new_blocks = [((r_pos[0]+1, r_pos[1]), (r_pos[0], r_pos[1]))]
        elif r_pos[1] == 0:
            new_blocks = [((r_pos[0], r_pos[1]+1), (r_pos[0], r_pos[1]))]
        new_blocks = tuple(new_blocks[:self.roadblocks])
        # print("new_blocks", new_blocks)
        return [GameState("play_b", c_pos, r_pos, new_blocks, state)]

def add_and_remove_edges(G, p_new_connection, p_remove_connection):
    '''
    for each node,
      add a new connection to random other node, with prob p_new_connection,
      remove a connection, with prob p_remove_connection

    operates on G in-place
    '''
    new_edges = []
    rem_edges = []
    # root_node = list(G.nodes)[0]

    for node in G.nodes():
        # find the other nodes this one is connected to
        connected = [to for (fr, to) in G.edges(node)]
        # and find the remainder of nodes, which are candidates for new edges
        unconnected = [n for n in G.nodes() if not n in connected]

        # probabilistically add a random edge
        if len(unconnected): # only try if new edge is possible
            if random.random() < p_new_connection:
                new = random.choice(unconnected)
                G.add_edge(node, new)
                # print("\tnew edge:\t {} -- {}".format(node, new))
                new_edges.append( (node, new) )
                # book-keeping, in case both add and remove done in same cycle
                unconnected.remove(new)
                connected.append(new)

        # probabilistically remove a random edge
        if len(connected) > 1: # only try if an edge exists to remove
            if random.random() < p_remove_connection:
                remove = random.choice(connected)
                G.remove_edge(node, remove)
                if nx.is_connected(G):
                    # print("\tedge removed:\t {} -- {}".format(node, remove))
                    rem_edges.append( (node, remove) )
                    # book-keeping, in case lists are important later?
                    connected.remove(remove)
                    unconnected.append(remove)
                else:
                    G.add_edge(node, remove)
    return rem_edges, new_edges

if __name__ == "__main__":
    n = 10
    G = nx.path_graph(n)
    print("XX", G.nodes, G )
    simulate_game(BaseStrategy(G, 1))


    G = nx.random_tree(n)



    n = 15

    G = nx.grid_2d_graph(n, n)#, periodic=False)

    # remove dimensional index
    # g = nx.to_numpy_array(G)
    # G = nx.from_numpy_array(g)

    # print("XX", G.nodes, G )

    simulate_game(GridStrategy(G, 1))
    print(len(G.edges()))
    a, b = add_and_remove_edges(G, 0.2, 0.2)
    print("removed", len(a), ", added", len(b))
    simulate_game(BaseStrategy(G, 1))
    print(len(G.edges()))
    # print(G)
    # print(list(G.edges))
