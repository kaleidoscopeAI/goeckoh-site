      } else {
          initSystem()
      }

      startButton.setOnClickListener {
          // (unchanged)
      }
  }

  private val metricsReceiver = object : BroadcastReceiver() {
      override fun onReceive(context: Context?, intent: Intent?) {
          val metrics = intent?.getStringExtra("metrics") ?: ""
          vadMetricsText.text = metrics
      }
  }

  override fun onDestroy() {
      localBroadcastManager.unregisterReceiver(metricsReceiver)
      super.onDestroy()
  }

  private fun initSystem() {
      // (unchanged)
  }
