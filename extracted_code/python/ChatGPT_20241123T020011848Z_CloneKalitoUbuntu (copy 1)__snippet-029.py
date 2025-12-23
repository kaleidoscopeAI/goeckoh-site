# knowledge_pool.py
from shared_utilities import EnhancedSharedKnowledgePool

class KnowledgePoolManager:
    """Manager for handling shared knowledge pools"""
    
    def __init__(self):
        self.shared_pool = EnhancedSharedKnowledgePool()
        
    def add_to_pool(self, pattern_data: Dict, confidence: float):
        """Add pattern data to the shared pool"""
        self.shared_pool.add_pattern(pattern_data, confidence)

