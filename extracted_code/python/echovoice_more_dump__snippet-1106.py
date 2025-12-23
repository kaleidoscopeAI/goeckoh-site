"""Background worker thread to process tasks"""
global current_task, running

while running:
    try:
        # Get task from queue
        task = task_queue.get(timeout=1.0)

        # Set as current task
        current_task = task
        current_task["status"] = "processing"

        # Process task
        if task["type"] == "ingest":
            worker_ingest(task)
        elif task["type"] == "mimic":
            worker_mimic(task)

        # Mark task as done
        task_queue.task_done()

        # Add to history
        task_history.append(task)

        # Limit history size
        if len(task_history) > 10:
            task_history.pop(0)

        # Clear current task
        current_task = None

    except queue.Empty:
        # No tasks in queue
        pass
    except Exception as e:
        logger.error(f"Error in worker thread: {str(e)}")

        # Update current task status
        if current_task:
            current_task["status"] = "error"
            current_task["error"] = str(e)

            # Add to history
            task_history.append(current_task)

            # Clear current task
            current_task = None

