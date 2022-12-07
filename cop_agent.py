
import random
import interaction_model as model

class CopBasic():
    def __init__(self, graph: model.Graph, pos) -> None:
        self.pos = pos
        self.graph = graph
    
    def make_move(self):
        # find best edge
        options = self.graph.nodes[self.pos].adjacent

        self.pos = random.choice(options)

    def find_corners(self):
        pass
        
