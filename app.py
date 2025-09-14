import json
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# --- 1. DATA LOADING ---
def load_documents(filepath="documents.json"):
    """Loads documents from a JSON file."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        return data["documents"]
    except FileNotFoundError:
        st.error(f"Error: The file {filepath} was not found.")
        return []
    except json.JSONDecodeError:
        st.error(f"Error: Could not decode JSON from {filepath}.")
        return []

# --- 2. CORE ML FUNCTIONS ---
def vectorize_text(documents):
    """Converts a list of documents into TF-IDF vectors."""
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(documents)
    return X, vectorizer

def perform_clustering(X, n_clusters):
    """Performs K-Means clustering on the given data."""
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    model.fit(X)
    return model.labels_

def reduce_dimensions(X):
    """Reduces the dimensionality of the data to 2D for plotting."""
    pca = PCA(n_components=2, random_state=42)
    X_reduced = pca.fit_transform(X.toarray())
    return X_reduced

# --- 3. VISUALIZATION ---
def plot_clusters(X_reduced, labels, n_clusters):
    """Plots the clustered documents."""
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(n_clusters):
        points = X_reduced[labels == i]
        ax.scatter(points[:, 0], points[:, 1], label=f"Cluster {i + 1}")
    ax.set_title("Document Clustering with K-Means", fontsize=16)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# --- 4. STREAMLIT UI ---
def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(page_title="Document Clustering System", layout="wide")
    
    st.title("📄 Document Clustering System")
    st.write("This application clusters text documents using TF-IDF and K-Means and visualizes the results.")

    documents = load_documents()
    
    if not documents:
        st.warning("No documents loaded. Please check the 'documents.json' file.")
        return

    # --- Sidebar for controls ---
    st.sidebar.header("Clustering Controls")
    n_clusters = st.sidebar.slider(
        "Select the number of clusters (K):", 
        min_value=2, 
        max_value=min(10, len(documents) -1), 
        value=3, 
        step=1
    )

    # --- Main processing and display ---
    if documents:
        X, vectorizer = vectorize_text(documents)
        labels = perform_clustering(X, n_clusters)
        X_reduced = reduce_dimensions(X)

        st.header("Cluster Visualization")
        plot_clusters(X_reduced, labels, n_clusters)

        st.header("Documents by Cluster")
        for i in range(n_clusters):
            with st.expander(f"Cluster {i + 1}"):
                cluster_docs = np.array(documents)[labels == i]
                for doc in cluster_docs:
                    st.write(f"- {doc}")

if __name__ == "__main__":
    main()