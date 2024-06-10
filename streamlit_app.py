import streamlit as st
from streamlit_agraph import Node, Edge, agraph, Config
from modules.data_processing import load_and_process_data
from modules.text_processing import preprocess_text
from modules.similarity import create_tfidf_matrix, calculate_similarity
import pandas as pd

# Streamlit page config
st.set_page_config(page_title="Interactive Network Map Dashboard", layout="wide")

def create_nodes_and_edges(data, similarity_matrix, threshold=0.5):
    node_dict = {}
    edges = []
    
    # Create unique nodes
    for index, row in data.iterrows():
        node_id = row['Speaker Name']
        if node_id not in node_dict:
            node_dict[node_id] = Node(id=node_id, label=f"{node_id} ({row['Speaker Title']})", size=300)
    
    # List of unique nodes
    nodes = list(node_dict.values())
    
    # Create edges based on the similarity matrix
    num_speakers = len(data)
    for i in range(num_speakers):
        for j in range(i + 1, num_speakers):  # To avoid self-looping and duplicate edges
            if similarity_matrix[i][j] > threshold:  # Only consider significant similarities
                src = data.iloc[i]['Speaker Name']
                tgt = data.iloc[j]['Speaker Name']
                if src in node_dict and tgt in node_dict:  # Ensure both nodes exist
                    edges.append(Edge(source=src, target=tgt, type="CURVE_SMOOTH"))

    return nodes, edges

def create_agraph_nodes_and_edges(G, pos):
    nodes = [Node(id=n, label=n, size=G.nodes[n]['size'], x=1000 * pos[n][0], y=1000 * pos[n][1]) for n in G.nodes]
    edges = [Edge(source=u, target=v, width=2 + 2 * G[u][v]['weight']) for u, v in G.edges]
    return nodes, edges

def find_similar_nodes(similarity_matrix, node_index, top_n=5):
    # Get all similarity scores for the selected node
    similarities = list(enumerate(similarity_matrix[node_index]))
    # Sort them by score in descending order, skipping the self-similarity at the first index
    sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)[1:top_n+1]
    return sorted_similarities

def main():
    st.title('Speaker Influence Network')

    # Load and process data
    data, similarity_matrix = load_and_process_data('data/Speakers_INK_Talks_Extracted.csv')
    if data is None or similarity_matrix is None:
        st.error("Failed to load or process data.")
        return

    # Node selection via dropdown
    node_options = {row['Speaker Name']: idx for idx, row in data.iterrows()}
    selected_node = st.selectbox('Select a Node:', list(node_options.keys()))

    # Finding the top similar nodes
    selected_node_index = node_options[selected_node]
    similar_nodes_indices = find_similar_nodes(similarity_matrix, selected_node_index)

    # Display similar nodes
    st.write("Top 5 Similar Nodes:")
    for idx, score in similar_nodes_indices:
        st.write(f"{data.iloc[idx]['Speaker Name']} - Similarity Score: {score}")

    # Visualize the selected node and similar nodes
    nodes, edges = create_nodes_and_edges(data, similarity_matrix, selected_node_index, similar_nodes_indices)
    config = Config(width=800, height=600, directed=False, nodeHighlightBehavior=True, highlightColor="#F7A7A6", collapsible=True)
    agraph(nodes=nodes, edges=edges, config=config)

def create_nodes_and_edges(data, similarity_matrix, selected_node_index, similar_nodes_indices, threshold=0.5):
    nodes = [Node(id=row['Speaker Name'], label=f"{row['Speaker Name']} ({row['Speaker Title']})", size=300)
             for index, row in data.iterrows()]
    edges = []
    for idx, score in similar_nodes_indices:
        if score > threshold:
            edges.append(Edge(source=data.iloc[selected_node_index]['Speaker Name'], target=data.iloc[idx]['Speaker Name'], type="CURVE_SMOOTH"))
    return nodes, edges

if __name__ == "__main__":
    main()