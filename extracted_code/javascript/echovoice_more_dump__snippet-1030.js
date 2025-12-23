"""
Worker function to mimic software

Args:
    task: Task information
"""
try:
    # Mimic software
    spec_files = task["spec_files"]
    target_language = task["target_language"]

    logger.info(f"Generating mimicked version in {target_language}...")

    # Run mimicry
    result = kaleidoscope.mimic_software(spec_files, target_language)

    # Update task with result
    task.update(result)
    task["status"] = result["status"]

    logger.info(f"Mimicry completed with status: {result['status']}")

except Exception as e:
    logger.error(f"Error in mimicry: {str(e)}")
    task["status"] = "error"
    task["error"] = str(e)

