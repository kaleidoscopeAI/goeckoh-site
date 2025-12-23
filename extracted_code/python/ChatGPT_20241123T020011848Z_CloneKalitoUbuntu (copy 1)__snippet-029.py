# core_demo.py
from knowledge_pool import KnowledgePoolManager
from reflection_analysis import SelfReflection

class CoreDemo:
    def __init__(self):
        self.knowledge_manager = KnowledgePoolManager()
        self.self_reflector = SelfReflection()

    def run_demo(self):
        # Sample pattern data to add to the pool
        sample_pattern = {
            "pattern_id": "001",
            "data": {"type": "text", "content": "AI pattern recognition"}
        }
        confidence = 0.75
        
        # Add pattern to knowledge pool
        self.knowledge_manager.add_to_pool(sample_pattern, confidence)
        
        # Perform self-reflection
        recent_actions = [{"action": "learn", "success": True}]
        reflection_output = self.self_reflector.reflect(recent_actions)
        
        # Display reflection insights
        print("Reflection Insights:", reflection_output)

if __name__ == "__main__":
    demo = CoreDemo()
    demo.run_demo()

