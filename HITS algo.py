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
