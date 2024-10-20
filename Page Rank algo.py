import networkx as nx
import matplotlib.pyplot as plt

# Create a directed graph
G = nx.DiGraph()

# Add edges (example)
edges = [(0, 2), (1, 0), (2, 1), (2, 3), (3, 2)]
G.add_edges_from(edges)

# Calculate PageRank
pagerank_scores = nx.pagerank(G, alpha=0.85)

# Visualize the graph
pos = nx.spring_layout(G)
plt.figure(figsize=(8, 6))
nx.draw(G, pos, with_labels=True, node_size=3000, node_color='skyblue', font_size=15, font_weight='bold')

# Display PageRank scores
for node, score in pagerank_scores.items():
    plt.text(pos[node][0], pos[node][1] + 0.1, s=f'{score:.2f}', bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center')

plt.title("PageRank Scores")
plt.show()

print("PageRank Scores:", pagerank_scores)

"""
Steps of the PageRank Algorithm
Initialize: Start by assigning each page an equal PageRank value, typically 1/N, where 𝑁
N is the total number of pages.

Iteration: For each page, update its PageRank based on the ranks of all pages that link to it, using the PageRank formula.

Convergence: Repeat the update process until the PageRank values for all pages stabilize (i.e., the change between iterations becomes very small).

Algorithm
Input: A set of pages and the links between them.
Initialization: Assign an initial PageRank of 1/𝑁 to each page.
Iteration:
For each page P, compute its new PageRank using the formula.
Distribute the PageRank of each page evenly to all the pages it links to.
Apply the damping factor.
Convergence: Continue iterating until PageRank values stabilize.
Output: The final PageRank values for each page.
Damping Factor
The damping factor d accounts for the probability that a user will continue following links on the web rather than starting over at a new page.
A typical value for d is 0.85, meaning there’s an 85% chance the user will follow links and a 15% chance they’ll jump to a random page.

Advantages
Simplicity: The algorithm is conceptually simple but effective in ranking web pages based on their relative importance.
Global Measure: PageRank provides a global measure of a web page’s importance, taking into account the entire link structure of the web.
Limitations
Link Spam: Webpages can manipulate the PageRank system through "link farms," where they artificially inflate the number of incoming links.
Personalization: PageRank does not consider user preferences or context, making it a general-purpose ranking system.
Applications
Web Search: PageRank is primarily used in search engines to rank pages.
Recommendation Systems: It can also be used in other applications like social networks or recommendation systems to identify important nodes.
Conclusion
PageRank is a foundational algorithm for web ranking that revolutionized how search engines operate by considering the link structure of the web.
Despite its simplicity, it has proven to be an effective way to rank pages based on their importance and continues to be a crucial component in 
modern search engine algorithms.
"""
