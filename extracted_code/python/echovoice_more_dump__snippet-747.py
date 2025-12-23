async def handle_query():
    data = await request.get_json()
    user_input = data.get("query", "")
    
    result = await cognitive_system.cognitive_cycle(user_input)
    
    return jsonify({
        "status": "success",
        "result": result,
        "system_id": cognitive_system.system_id
    })

