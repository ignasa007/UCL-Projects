import networkx as nx
from Q5a import mc

# Create a list of edges in the Markov chain graph
edges = list()
for beta in mc.symbols:
    for alpha in mc.symbols:
        if mc.transition_prob(beta, alpha) > 0:
            edges.append(tuple((beta, alpha)))

# Create the directed graph
G = nx.DiGraph()
G.add_edges_from(edges)

# Calculate the number of SCCs
print(f'Number of strongly connected components = {len(list(nx.strongly_connected_components(G)))}')
# Number of strongly connected components = 1