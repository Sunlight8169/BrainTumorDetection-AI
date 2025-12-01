from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

class NLPProcessor:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        # Load sentence transformer model
        print("Loading NLP model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.stop_words = set(stopwords.words('english'))
        print("NLP model loaded successfully!")
    
    def clean_text(self, text):
        """Clean and preprocess text"""
        # Convert to lowercase
        text = text.lower()
        # Remove special characters
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        # Remove extra whitespace
        text = ' '.join(text.split())
        return text
    
    def remove_stopwords(self, text):
        """Remove stopwords from text"""
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in self.stop_words]
        return ' '.join(filtered_text)
    
    def extract_keywords(self, text):
        """Extract important keywords"""
        cleaned = self.clean_text(text)
        without_stopwords = self.remove_stopwords(cleaned)
        return without_stopwords.split()
    
    def get_embedding(self, text):
        """Generate sentence embedding"""
        return self.model.encode(text)
    
    def calculate_similarity(self, text1, text2):
        """Calculate cosine similarity between two texts"""
        embedding1 = self.get_embedding(text1).reshape(1, -1)
        embedding2 = self.get_embedding(text2).reshape(1, -1)
        return cosine_similarity(embedding1, embedding2)[0][0]