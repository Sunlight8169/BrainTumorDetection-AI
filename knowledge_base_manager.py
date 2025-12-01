import json
import os

class KnowledgeBaseManager:
    def __init__(self, kb_path='data/knowledge_base.json'):
        self.kb_path = kb_path
        self.knowledge_base = self.load_knowledge_base()
    
    def load_knowledge_base(self):
        """Load knowledge base from JSON file"""
        try:
            # Try relative path first
            if os.path.exists(self.kb_path):
                with open(self.kb_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            
            # Try absolute path
            script_dir = os.path.dirname(os.path.abspath(__file__))
            kb_path = os.path.join(script_dir, '..', self.kb_path)
            
            with open(kb_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Knowledge base file not found: {self.kb_path}")
            print(f"Current directory: {os.getcwd()}")
            return {"questions": []}
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            return {"questions": []}
    
    def get_all_questions(self):
        """Get all questions from knowledge base"""
        return self.knowledge_base.get('questions', [])
    
    def get_question_by_id(self, question_id):
        """Get specific question by ID"""
        for q in self.knowledge_base.get('questions', []):
            if q['id'] == question_id:
                return q
        return None
    
    def get_questions_by_category(self, category):
        """Get questions filtered by category"""
        return [q for q in self.knowledge_base.get('questions', []) 
                if q.get('category') == category]
    
    def search_by_keywords(self, keywords):
        """Search questions by keywords"""
        results = []
        for q in self.knowledge_base.get('questions', []):
            q_keywords = [kw.lower() for kw in q.get('keywords', [])]
            if any(keyword.lower() in q_keywords for keyword in keywords):
                results.append(q)
        return results