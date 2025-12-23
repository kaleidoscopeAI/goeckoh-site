def allowed_file(filename):
    """Check if a file has an allowed extension"""
    # Allow most executable and code file extensions
    allowed_extensions = {
        'exe', 'dll', 'so', 'dylib',  # Binaries
        'js', 'mjs',                  # JavaScript
        'py',                         # Python
        'c', 'cpp', 'h', 'hpp',       # C/C++
        'java', 'class', 'jar',       # Java
        'go',                         # Go
        'rs',                         # Rust
        'php',                        # PHP
        'rb',                         # Ruby
        'cs',                         # C#
        'asm', 's'                    # Assembly
    }
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def worker_loop():
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

def worker_ingest(task):
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

def worker_mimic(task):
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

