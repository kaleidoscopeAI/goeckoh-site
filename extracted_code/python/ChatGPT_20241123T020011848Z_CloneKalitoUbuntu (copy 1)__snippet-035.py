# confidence_calculation_demo.py
from multi_factor_confidence import MultiFactorConfidence

# Initialize confidence calculator
confidence_calculator = MultiFactorConfidence()

# Sample pattern and context
pattern_data = {'pattern': 'Sample'}
context_data = {'environment': 'test_env'}

# Calculate confidence
confidence = confidence_calculator.calculate_confidence(pattern_data, context_data)
print("Calculated Confidence:", confidence)

