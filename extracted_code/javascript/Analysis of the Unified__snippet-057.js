Specializations: Gears are categorized into pattern_recognition, data_transformation, memory_integration, decision_making, and learning_optimization.

Suitability Routing: Data is routed to the most appropriate gear by calculating a score that is weighted by:

    Specialization Match: A multiplier (e.g., ×1.5) is applied if the data structure (e.g., values, structure, evidence) matches the gear's specialization.

    Activity Factor: The score is scaled by an inverse function of time since the gear was last active (1.0+time_since_active/36001​) to promote distributed processing and prevent bottlenecks.

