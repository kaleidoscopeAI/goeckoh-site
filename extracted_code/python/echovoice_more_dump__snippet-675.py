async def get_status():
    metrics = organic_ai.metrics
    return jsonify({
        "health": metrics.health,
        "coherence": metrics.coherence,
        "complexity": metrics.complexity,
        "emergence": metrics.emergence_level,
        "energy_efficiency": metrics.energy_efficiency
    })

