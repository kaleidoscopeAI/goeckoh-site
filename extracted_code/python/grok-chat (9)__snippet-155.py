  """
  Generate the llms.txt file for a given repository.
  """

  if log_level: # Override log level from cmd line
      os.environ["LOG_LEVEL"] = log_level.upper()
      console.print(f":exclamation: Overriding LOG_LEVEL: [bold cyan]
