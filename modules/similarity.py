from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def create_tfidf_matrix(texts):
    """
    Create a TF-IDF matrix for a list of texts.

    Args:
    texts (list): List of preprocessed text documents.

    Returns:
    tuple: TF-IDF matrix and the vectorizer used.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    return tfidf_matrix, vectorizer

def calculate_similarity(tfidf_matrix):
    """
    Calculate cosine similarity matrix from a TF-IDF matrix.

    Args:
    tfidf_matrix (sparse matrix): TF-IDF matrix of text documents.

    Returns:
    ndarray: Cosine similarity matrix.
    """
    return cosine_similarity(tfidf_matrix)
