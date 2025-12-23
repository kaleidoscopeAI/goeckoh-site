108      private fun initSystem() {
109 -        statusText.text = "System Initialized"
109 +        statusText.text = "Preparing models..."
110 +        try {
111 +            AssetInstaller.installAll(applicationContext, statusText)
112 +            statusText.text = "System Initialized"
113 +        } catch (e: Exception) {
114 +            statusText.text = "Init failed: ${e.message}"
115 +        }
116      }

