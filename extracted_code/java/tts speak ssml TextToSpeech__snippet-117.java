      updateVadFromPrefs()
  }

  private fun updateVadFromPrefs() {
      vadMode = sharedPrefs.getInt("vad_mode", 1) // 0 NORMAL, 1 LOW, 2
