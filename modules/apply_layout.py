import networkx as nx

def apply_layout(data, similarity_matrix, threshold=0.5):
    G = nx.Graph()
    for index, row in data.iterrows():
        G.add_node(row['Speaker Name'], size=300 + 10 * row['degree'])  # Increase size based on degree

    for i in range(len(data)):
        for j in range(i + 1, len(data)):
            if similarity_matrix[i][j] > threshold:
                G.add_edge(data.iloc[i]['Speaker Name'], data.iloc[j]['Speaker Name'], weight=similarity_matrix[i][j])

    # Apply a force-directed layout
    pos = nx.spring_layout(G, weight='weight')  # Use edge weight to influence layout
    return G, pos
