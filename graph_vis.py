from turtle import pos
from pyvis.network import Network
import networkx as nx
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import random
import graph_generate
from matplotlib.widgets import Button

# nx_graph = nx.cycle_graph(10)
# nx_graph.nodes[1]['title'] = 'Number 1'
# nx_graph.nodes[1]['group'] = 1
# nx_graph.nodes[3]['title'] = 'I belong to a different group!'
# nx_graph.nodes[3]['group'] = 10
# nx_graph.add_node(20, size=20, title='couple', group=2)
# nx_graph.add_node(21, size=15, title='couple', group=2)
# nx_graph.add_edge(20, 21, weight=5)
# nx_graph.add_node(25, size=25, label='lonely', title='lonely node', group=3)

def draw_network(G: nx.Graph):
    # nt = Network('500px', '500px')
    nt = Network()
    # populates the nodes and edges data structures
    nt.from_nx(G)
    nt.show('nx.html')



draw_config = {
    "node_size": 85,
    "with_labels": True,
    "font_size": 8,
    "linewidths": 0,
    # "width": 2,
    # "node_color": "silver",
}

G = nx.Graph()
######## GRAPH TOPOLOGY ########
# graph_generate.sample(G)
# graph_generate.create_clique(G, 9)
# graph_generate.create_clique(G, 9)
graph_generate.create_path(G, 9, True)
# graph_generate.create_grid_square(G, 5, 5, 1)
# G.remove_edge(7, 2)
# G.remove_edge(8, 2)
G.add_edge(1, 14)
G.add_edge(1, 15)
G.add_edge(14, 16)
# G.add_edge(1, 25)
# graph_generate.create_grid_hex(G, 7, 4)
graph_generate.find_corners(G)


######## LAYOUTS ########
# pos_layout = nx.shell_layout(G)
# pos_layout = nx.circular_layout(G)
# pos_layout = nx.spring_layout(G)
# pos_layout = nx.spring_layout(G, scale=0.1, k=0.01)
# pos_layout = nx.fruchterman_reingold_layout(G, seed=1)
# pos_layout = nx.planar_layout(G)
# pos_layout = nx.spring_layout(G, pos=pos_layout, iterations=1250)
# pos_layout = nx.fruchterman_reingold_layout(G, seed=1, pos=pos_layout, iterations=1500)
pos_layout = nx.fruchterman_reingold_layout(G, seed=1, iterations=500)


# Animation funciton
def animate(event):
    global G
    # colors = ['r', 'b', 'g', 'y', 'w', 'm']
    colors = ['y', 'w']
    # nx.draw_circular(G, node_color=[random.choice(colors) for j in range(9)])
    # nx.draw_kamada_kawai(G, node_color=[random.choice(colors) for j in range(9)])
    nx.draw(G, pos=pos_layout, ax=ax, node_color=[random.choice(colors) if G.nodes[j].get("group",0)!=2 else "r" for j in G.nodes()], **draw_config)
    plt.draw()

fig, ax = plt.subplots()
fig.subplots_adjust(bottom=0.2)

# nx.draw_circular(G)
# nx.draw_kamada_kawai(G)
nx.draw(G, pos=pos_layout, ax=ax, **draw_config)
fig = plt.gcf()

# Animator call
# anim = animation.FuncAnimation(fig, animate, frames=20, interval=2000)
# anim = animation.FuncAnimation(fig, animate, frames=20)

axprev = fig.add_axes([0.7, 0.05, 0.1, 0.075])
axnext = fig.add_axes([0.81, 0.05, 0.1, 0.075])
bnext = Button(axnext, 'Next')
bnext.on_clicked(animate)
bprev = Button(axprev, 'Previous')
bprev.on_clicked(animate)

# draw_network(G)
plt.show()