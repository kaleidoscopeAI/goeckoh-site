• The Sherpa error indicates missing assets causing early failure that prevents Vosk fallback from initializing properly, likely
  due to silent or suppressed Vosk import/load errors. To fix this, I'll bypass Sherpa and prioritize Vosk when an environment
  flag is set, avoiding helper caching and ensuring Vosk loads directly.

