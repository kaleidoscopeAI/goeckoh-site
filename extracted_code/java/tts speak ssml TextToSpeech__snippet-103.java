      val buffer = ByteArray(bufferSize)
      var silenceStart = 0L
      val startTime = System.currentTimeMillis()
      while (true) {
          val read = audioRecord?.read(buffer, 0, buffer.size) ?: 0
          if (read > 0) {
              audioBuffer.write(buffer, 0, read)
              val shortBuffer = ShortArray(read / 2)
              ByteBuffer.wrap(buffer, 0,
