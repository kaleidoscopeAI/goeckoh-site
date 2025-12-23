def mimic_software():
    """Handle request to mimic software"""
    data = request.json
    
    # Check required fields
    if not data or 'language' not in data or 'task_id' not in data:
        return jsonify({"error": "Missing required fields"}), 400
    
    # Get source task
    source_task = None
    for task in task_history:
        if task.get("id") == data['task_id']:
            source_task = task
            break
    
    if not source_task or source_task["type"] != "ingest" or source_task["status"] != "completed":
        return jsonify({"error": "Source task not found or not completed"}), 400
    
    # Check if we have specification files
    if "spec_files" not in source_task or not source_task["spec_files"]:
        return jsonify({"error": "No specification files available"}), 400
    
    # Validate target language
    target_language = data['language'].lower()
    valid_languages = ["python", "javascript", "c", "cpp", "c++", "java"]
    
    if target_language not in valid_languages:
        return jsonify({"error": f"Unsupported language: {target_language}"}), 400
    
    # Map language aliases
    if target_language in ["c++", "cpp"]:
        target_language = "cpp"
    
    # Create task
    task_id = str(uuid.uuid4())
    task = {
        "id": task_id,
        "type": "mimic",
        "source_task_id": data['task_id'],
        "spec_files": source_task["spec_files"],
        "target_language": target_language,
        "status": "queued",
        "timestamp": time.time()
    }
    
    # Add to queue
    task_queue.put(task)
    
    return jsonify({
        "status": "success",
        "message": f"Queued mimicry in {target_language}",
        "task_id": task_id
    })

