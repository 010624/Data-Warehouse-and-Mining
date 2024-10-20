import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add edges (example)
edges = [(0, 2), (1, 0), (2, 1), (2, 3), (3, 2)]
G.add_edges_from(edges)

# Calculate HITS
hits_scores = nx.hits(G)

# Visualize the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, font_weight='bold')

# Display HITS scores
authorities, hubs = hits_scores
for node, score in authorities.items():
    plt.text(pos[node][0], pos[node][1] + 0.1, s=f'A:{score:.2f} H:{hubs[node]:.2f}', bbox=dict(facecolor='green', alpha=0.5), horizontalalignment='center')

plt.title("HITS Scores")
plt.show()

print("Authority Scores:", authorities)
print("Hub Scores:", hubs)


# The HITS (Hyperlink-Induced Topic Search) algorithm, also known as the Hubs and Authorities algorithm, is a graph-based link analysis algorithm used to rank web pages or nodes in a directed graph. 
#The algorithm identifies two key types of nodes:

# Authorities: Nodes with valuable information that are linked to by other nodes (these are "expert" nodes in some context).
# Hubs: Nodes that act as directories, pointing to authoritative nodes (these are "reference" nodes).
# In essence, a good hub points to many good authorities, and a good authority is linked to by many good hubs.

# The HITS algorithm works by iterating through the graph and adjusting the scores of each node until convergence. This is done by following these steps:

# Initialize the hub score and authority score of all nodes to 1.
# Update each node's authority score to be the sum of the hub scores of the nodes linking to it.
# Update each node's hub score to be the sum of the authority scores of the nodes it links to.
# Normalize the scores to prevent them from growing indefinitely.
