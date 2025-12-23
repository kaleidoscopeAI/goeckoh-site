"""Interactive chatbot for the Kaleidoscope AI system"""

def __init__(self, work_dir: str = None):
    """
    Initialize the chatbot

    Args:
        work_dir: Working directory for the Kaleidoscope core
    """
    self.work_dir = work_dir or os.path.join(os.getcwd(), "kaleidoscope_workdir")
    self.kaleidoscope = KaleidoscopeCore(work_dir=self.work_dir)
    self.current_task = None
    self.task_queue = queue.Queue()
    self.worker_thread = None
    self.running = False
    self.session_history = []

    # Welcome message components
    self.welcome_message = [
        "Welcome to Kaleidoscope AI Chatbot!",
        "I can help you analyze, decompile, and mimic software through a conversational interface.",
        "Type 'help' to see available commands or 'exit' to quit."
    ]

    # Command handlers
    self.commands = {
        "help": self._handle_help,
        "ingest": self._handle_ingest,
        "status": self._handle_status,
        "list": self._handle_list,
        "mimic": self._handle_mimic,
        "info": self._handle_info,
        "clear": self._handle_clear,
        "exit": self._handle_exit
    }

def start(self):
    """Start the chatbot"""
    self.running = True

    # Start worker thread
    self.worker_thread = threading.Thread(target=self._worker_loop)
    self.worker_thread.daemon = True
    self.worker_thread.start()

    # Print welcome message
    for line in self.welcome_message:
        print(line)
    print()

    # Main interaction loop
    while self.running:
        try:
            # Get user input
            user_input = input("> ").strip()

            # Process user input
            if not user_input:
                continue

            # Record in history
            self.session_history.append({"role": "user", "content": user_input})

            # Process command
            self._process_input(user_input)

        except KeyboardInterrupt:
            print("\nExiting...")
            self.running = False
            break
        except Exception as e:
            logger.error(f"Error processing input: {str(e)}")
            print(f"Sorry, an error occurred: {str(e)}")

def _process_input(self, user_input: str):
    """
    Process user input and execute appropriate command

    Args:
        user_input: User input string
    """
    # Split into command and arguments
    parts = user_input.split(maxsplit=1)
    command = parts[0].lower()
    args = parts[1] if len(parts) > 1 else ""

    # Check if command exists
    if command in self.commands:
        # Execute command handler
        response = self.commands[command](args)

        # Record in history
        self.session_history.append({"role": "assistant", "content": response})

        # Print response
        print(response)
    else:
        # Handle unknown command
        response = f"Unknown command: '{command}'. Type 'help' to see available commands."
        self.session_history.append({"role": "assistant", "content": response})
        print(response)

def _worker_loop(self):
    """Background worker thread to process tasks"""
    while self.running:
        try:
            # Get task from queue
            task = self.task_queue.get(timeout=1.0)

            # Process task
            if task["type"] == "ingest":
                self._worker_ingest(task)
            elif task["type"] == "mimic":
                self._worker_mimic(task)

            # Mark task as done
            self.task_queue.task_done()

        except queue.Empty:
            # No tasks in queue
            pass
        except Exception as e:
            logger.error(f"Error in worker thread: {str(e)}")

            # Update current task status
            if self.current_task:
                self.current_task["status"] = "error"
                self.current_task["error"] = str(e)
                self.current_task = None

def _worker_ingest(self, task: Dict[str, Any]):
    """
    Worker function to ingest software

    Args:
        task: Task information
    """
    try:
        # Set as current task
        self.current_task = task
        self.current_task["status"] = "processing"

        # Ingest software
        file_path = task["file_path"]

        print(f"Ingesting {file_path}... This may take a while.")

        # Run ingestion
        result = self.kaleidoscope.ingest_software(file_path)

        # Update task with result
        task.update(result)
        task["status"] = result["status"]

        if result["status"] == "completed":
            print(f"\nIngestion completed successfully!")
            print(f"- Decompiled files: {len(result['decompiled_files'])}")
            print(f"- Specification files: {len(result['spec_files'])}")
            print(f"- Reconstructed files: {len(result['reconstructed_files'])}")
        else:
            print(f"\nIngestion failed: {result['status']}")
            if "error" in result:
                print(f"Error: {result['error']}")

    except Exception as e:
        logger.error(f"Error in ingestion: {str(e)}")
        task["status"] = "error"
        task["error"] = str(e)
        print(f"\nError during ingestion: {str(e)}")
    finally:
        # Clear current task
        self.current_task = None

def _worker_mimic(self, task: Dict[str, Any]):
    """
    Worker function to mimic software

    Args:
        task: Task information
    """
    try:
        # Set as current task
        self.current_task = task
        self.current_task["status"] = "processing"

        # Mimic software
        spec_files = task["spec_files"]
        target_language = task["target_language"]

        print(f"Generating mimicked version in {target_language}... This may take a while.")

        # Run mimicry
        result = self.kaleidoscope.mimic_software(spec_files, target_language)

        # Update task with result
        task.update(result)
        task["status"] = result["status"]

        if result["status"] == "completed":
            print(f"\nMimicry completed successfully!")
            print(f"- Generated {len(result['mimicked_files'])} files")
            print(f"- Output directory: {result['mimicked_dir']}")
        else:
            print(f"\nMimicry failed: {result['status']}")
            if "error" in result:
                print(f"Error: {result['error']}")

    except Exception as e:
        logger.error(f"Error in mimicry: {str(e)}")
        task["status"] = "error"
        task["error"] = str(e)
        print(f"\nError during mimicry: {str(e)}")
    finally:
        # Clear current task
        self.current_task = None

def _handle_help(self, args: str) -> str:
    """
    Handle 'help' command

    Args:
        args: Command arguments

    Returns:
        Response message
    """
    return """
