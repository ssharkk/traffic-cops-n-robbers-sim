# from pyvis.network import Network
import networkx as nx
from collections import namedtuple
import pprint

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
        new_c_pos = len(self.source_graph)//2
        return [GameState("place_c", new_c_pos, r_pos, blocked_edges, state)]

    def get_robber_start_position(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state

        return [GameState("place_r", c_pos, new_r_pos, blocked_edges, state) for new_r_pos in range(len(self.source_graph)) if new_r_pos != c_pos]

    def play_cop(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        cur_dist = len(nx.shortest_path(self.source_graph, c_pos, r_pos))
        moves = [new_c_pos for new_c_pos in nx.neighbors(self.source_graph, c_pos) if len(nx.shortest_path(self.source_graph, new_c_pos, r_pos)) < cur_dist]
        plays = [GameState("play_c", new_c_pos, r_pos, blocked_edges, state) for new_c_pos in moves]
        # print("cop", c_pos, moves)
        return plays

    def play_robber(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        cur_dist = len(nx.shortest_path(self.source_graph, c_pos, r_pos))
        for a, b in blocked_edges:
            # print("remove", a, b)
            self.source_graph.remove_edge(a, b)
        moves = [new_r_pos for new_r_pos in nx.neighbors(self.source_graph, r_pos)]
        for a, b in blocked_edges:
            self.source_graph.add_edge(a, b)
            # print("add", a, b)
        moves = [(len(nx.shortest_path(self.source_graph, c_pos, new_r_pos)), new_r_pos) for new_r_pos in moves]
        best_loss = max(moves)[0]
        moves = [move[1] for move in moves if move[0] == best_loss]
        plays = [GameState("play_r", c_pos, new_r_pos, blocked_edges, state) for new_r_pos in moves]
        # print("robber", r_pos, moves, list(nx.neighbors(self.source_graph, r_pos)))
        return plays

    def play_roadblocks(self, state):
        turn, c_pos, r_pos, blocked_edges, prev_state = state
        cur_dist = len(nx.shortest_path(self.source_graph, c_pos, r_pos))
        new_blocks = [(r_pos, v) for v in nx.neighbors(self.source_graph, r_pos) if len(nx.shortest_path(self.source_graph, c_pos, v)) > cur_dist][:self.roadblocks]
        # print("roadblock", len(new_blocks))
        new_blocks = tuple(new_blocks)
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

def simulate_game(G_strategy):
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
        # print("State:")
        # print(G_strategy.eval_winner(cur_states[-1]))
        # print(cur_states[-1])
        while len(cur_states) > 0 and G_strategy.eval_winner(cur_states[-1]):
            win_state = cur_states.pop()
            all_traces.append(win_state)
            trace_capturetimes.append(G_strategy.capture_time(win_state)[0])
            # print(f"WIN({ G_strategy.capture_time(win_state) }) \t", win_state.r_pos, win_state.c_pos, win_state.turn[-1])
            continue
            # trace = win_state
            # turns = 0
            # while win_state.prev_state is not None:
            #     # print(type(win_state), win_state)
            #     # tag = (win_state.c_pos, win_state.r_pos, win_state.prev_state.c_pos, win_state.prev_state.r_pos)
            #     tag = (win_state.c_pos, win_state.r_pos)
            #     print("write", tag)
            #     if tag in results and results[tag] != turns:
            #         print("UPDATING", tag, results[tag], "to", turns)
            #     results[tag] = turns
            #     # results[win_state] = turns
            #     ends += 1
            #     prev_win_state = win_state.prev_state
            #     if win_state.turn == "play_c":
            #         turns += 1
            #     if prev_win_state.prev_state is None:
            #         print("PREV NONE", prev_win_state)
            #         break
            #     else:
            #         # next_tag = (win_state.prev_state.c_pos, win_state.prev_state.r_pos, win_state.prev_state.prev_state.c_pos, win_state.prev_state.prev_state.r_pos)
            #         next_tag = (win_state.prev_state.c_pos, win_state.prev_state.r_pos)
            #         if next_tag in results:
            #             # print("BREAKS", tag, next_tag)
            #             if win_state.turn in ("play_c") and   results[next_tag] >= turns:
            #                 print("break as ", win_state.turn, tag, next_tag)
            #                 break
            #             elif win_state.turn in ("play_r") and results[next_tag] <= turns:
            #                 print("break as ", win_state.turn, tag, next_tag)
            #                 break
            #     win_state = prev_win_state
            #     if win_state.turn == "play_b":
            #         win_state = prev_win_state
            #
            # if prev_win_state.turn == "init":
            #     print("hello")
            #     best_trace = trace
            # pprint.pprint(results)

            # print(trace)
        # f += 1
        # if f > 15: break

    # # print(best_trace)
    # print("result below", len(results), ends)
    # pprint.pprint(results)
    # print("result above")
    # print(cur_states)
    print(len(all_traces), "WAYS FOR COP TO WIN!", sum(trace_capturetimes) / len(all_traces))


# print({("a", 1) : 1, ("a", []): 2})
n = 30
G = nx.path_graph(n)
simulate_game(BaseStrategy(G, 1))

G = nx.random_tree(n)
simulate_game(BaseStrategy(G, 1))

print(G)
print(list(G.edges))
