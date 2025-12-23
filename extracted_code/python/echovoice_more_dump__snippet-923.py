async def status():
    return jsonify({
        'state': 'idle',
        'nodes': len(core.nodes),
        'memory_size': len(core.memory.meta),
        'knowledge_points': len(core.knowledge_points)
    })

