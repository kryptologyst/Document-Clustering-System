# Document Clustering System

This project is an interactive web application that clusters text documents using TF-IDF for vectorization and K-Means for clustering. The results are visualized in a 2D plot using PCA for dimensionality reduction.

## Features

- **Interactive UI**: Built with Streamlit for a user-friendly experience.
- **Dynamic Clustering**: Adjust the number of clusters (K) in real-time.
- **Data-Driven**: Loads documents from an external `documents.json` file.
- **Clear Visualization**: Displays clustered documents in a scatter plot.
- **Easy to Explore**: View the documents belonging to each cluster.

## Setup and Installation

1.  **Clone the repository (or download the source code).**

2.  **Create and activate a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

Once the setup is complete, run the Streamlit application with the following command:

```bash
streamlit run app.py
```

The application will open in your default web browser.
# Document-Clustering-System
