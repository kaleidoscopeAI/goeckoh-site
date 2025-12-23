# knowledge_pool_demo.py
from knowledge_pool import EnhancedSharedKnowledgePool

# Initialize the knowledge pool
pool = EnhancedSharedKnowledgePool()

# Sample pattern data for testing
pattern_data = {
    'pattern': 'Sample Pattern',
    'context': 'Example Context'
}

# Add patterns with confidence levels
pool.add_pattern(pattern_data, confidence=0.85)
pool.add_pattern({'pattern': 'Another Pattern'}, confidence=0.7)

# Output the patterns to verify they were added
print("Patterns in pool:", pool.patterns)

