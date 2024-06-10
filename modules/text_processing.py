import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def preprocess_text(text):
    """
    Process text by lowering case, removing stopwords, and applying lemmatization.

    Args:
    text (str): Text to process.

    Returns:
    str: Processed text.
    """
    # Check if text is not a string
    if not isinstance(text, str):
        # If text is NaN or not a string, return an empty string
        return ""

    # Initialize NLP tools
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    # Normalize text
    text = text.lower()
    # Tokenize text
    words = word_tokenize(text)
    # Remove stopwords and non-alphanumeric characters, and lemmatize
    filtered_words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]

    return ' '.join(filtered_words)
