import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_and_process_data(file_path):
    """
    Load data from a CSV file, preprocess it, and compute a similarity matrix.

    Args:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame, np.array: Loaded and cleaned data, Similarity matrix
    """
    try:
        # Load data
        data = pd.read_csv(file_path)
        # Handle missing values
        data['Speaker Name'].fillna("", inplace=True)
        data['Speaker Title'].fillna("", inplace=True)
        data['Speaker Description'].fillna("", inplace=True)

        # Ensure there are no empty names as they will be used as IDs in the graph
        if data['Speaker Name'].isnull().any() or data['Speaker Name'].eq("").any():
            raise ValueError("Speaker Name column contains null or empty values, which are not allowed.")

        # Preprocess descriptions for similarity calculation
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(data['Speaker Description'])

        # Calculate the cosine similarity matrix
        similarity_matrix = cosine_similarity(tfidf_matrix)

        return data, similarity_matrix
    except FileNotFoundError:
        print(f"Error: The file {file_path} does not exist.")
        return None, None
    except pd.errors.EmptyDataError:
        print(f"No data: Check if the file is empty {file_path}.")
        return None, None
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        return None, None
