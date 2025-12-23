      setup_logger(current_config.agent_name)

  if max_files_to_process: # Override max files to process from cmd line
      os.environ["MAX_FILES_TO_PROCESS"] = str(max_files_to_process)
      current_config.invalidate()
      console.print(f":exclamation: Overriding MAX_FILES_TO_PROCESS: [bold
