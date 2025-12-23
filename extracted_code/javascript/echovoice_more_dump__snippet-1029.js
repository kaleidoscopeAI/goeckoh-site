"""
Worker function to ingest software

Args:
    task: Task information
"""
try:
    # Ingest software
    file_path = task["file_path"]

    logger.info(f"Ingesting {file_path}...")

    # Run ingestion
    result = kaleidoscope.ingest_software(file_path)

    # Update task with result
    task.update(result)
    task["status"] = result["status"]

    logger.info(f"Ingestion completed with status: {result['status']}")

except Exception as e:
    logger.error(f"Error in ingestion: {str(e)}")
    task["status"] = "error"
    task["error"] = str(e)

