from typing import List
import pyvis.network

class Node():
    def __init__(self, adjacent) -> None:
        self.adjacent = adjacent
        self.blocked = []

class Graph():

    # nodes: 
    def __init__(self, nodes: dict):
        self.blocked_edges = {}
        self.score = 0
        self.cops = []
        self.robbers = []
        self.nodes = nodes
    
    def spawn_cop(self):
        pass

    def spawn_robber(self):
        pass

    def block_path(self, edge):
        if edge in self.blocked_edges:
            return False
        else:
            self.blocked_edges[edge] = -1
            return True

    def release_path(self, edge):
        if edge not in self.blocked_edges:
            return False
        else:
            del self.blocked_edges[edge]
            return True

    def eval_turn(self):
        score += sum(self.blocked_edges.values())
        for r in self.robbers:
            for c in self.cops:
                if r.pos == c.pos:
                    return True # cop wins
        return False
