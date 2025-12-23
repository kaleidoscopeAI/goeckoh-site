def get_status():
    """Get status of tasks"""
    # Prepare current task info if available
    current = None
    if current_task:
        current = {
            "id": current_task.get("id"),
            "type": current_task.get("type"),
            "status": current_task.get("status"),
            "file_name": current_task.get("file_name") if "file_name" in current_task else None,
            "target_language": current_task.get("target_language") if "target_language" in current_task else None,
            "timestamp": current_task.get("timestamp")
        }
    
    # Prepare task history info
    history = []
    for task in task_history:
        task_info = {
            "id": task.get("id"),
            "type": task.get("type"),
            "status": task.get("status"),
            "file_name": task.get("file_name") if "file_name" in task else None,
            "target_language": task.get("target_language") if "target_language" in task else None,
            "timestamp": task.get("timestamp")
        }
        
        # Add success counts for completed tasks
        if task.get("status") == "completed":
            if task.get("type") == "ingest":
                task_info["decompiled_count"] = len(task.get("decompiled_files", []))
                task_info["spec_count"] = len(task.get("spec_files", []))
                task_info["reconstructed_count"] = len(task.get("reconstructed_files", []))
            elif task.get("type") == "mimic":
                task_info["mimicked_count"] = len(task.get("mimicked_files", []))
                task_info["mimicked_dir"] = task.get("mimicked_dir")
        
        history.append(task_info)
    
    return jsonify({
        "current": current,
        "history": history
    })

