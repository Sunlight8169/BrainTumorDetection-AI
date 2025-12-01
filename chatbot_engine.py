from src.nlp_processor import NLPProcessor
from src.knowledge_base_manager import KnowledgeBaseManager
import numpy as np

class ChatbotEngine:
    def __init__(self, similarity_threshold=0.5):
        print("Initializing Chatbot Engine...")
        self.nlp = NLPProcessor()
        self.kb_manager = KnowledgeBaseManager()
        self.similarity_threshold = similarity_threshold
        
        # Precompute embeddings for all questions
        self.question_embeddings = {}
        self._precompute_embeddings()
        print(f"✅ Chatbot ready! Loaded {len(self.question_embeddings)} questions.")
    
    def _precompute_embeddings(self):
        """Precompute embeddings for all questions in KB"""
        questions = self.kb_manager.get_all_questions()
        if not questions:
            print("⚠️ Warning: No questions found in knowledge base!")
            return
            
        for q in questions:
            question_text = q['question']
            self.question_embeddings[q['id']] = self.nlp.get_embedding(question_text)
    
    def find_best_match(self, user_query):
        """Find the best matching question from knowledge base"""
        if not self.question_embeddings:
            return None, 0.0
            
        # Clean user query
        cleaned_query = self.nlp.clean_text(user_query)
        query_embedding = self.nlp.get_embedding(cleaned_query)
        
        best_match = None
        best_score = 0
        
        # Calculate similarity with all questions
        for q in self.kb_manager.get_all_questions():
            q_embedding = self.question_embeddings[q['id']]
            
            # Calculate cosine similarity
            similarity = np.dot(query_embedding, q_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(q_embedding)
            )
            
            if similarity > best_score:
                best_score = similarity
                best_match = q
        
        # Return match if above threshold
        if best_score >= self.similarity_threshold:
            return best_match, best_score
        else:
            return None, best_score
    
    def get_response(self, user_query):
        """Get chatbot response for user query"""
        match, score = self.find_best_match(user_query)
        
        if match:
            return {
                'answer': match['answer'],
                'confidence': round(score * 100, 2),
                'category': match.get('category', 'general'),
                'matched': True
            }
        else:
            return {
                'answer': "I'm sorry, I don't have specific information about that. Please consult a medical professional for accurate diagnosis and treatment advice. You can also try:\n- Rephrasing your question\n- Using the quick question buttons in the sidebar\n- Uploading an MRI scan for AI analysis",
                'confidence': round(score * 100, 2),
                'category': 'unknown',
                'matched': False
            }