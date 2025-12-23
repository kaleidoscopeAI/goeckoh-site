    for (let i = 0; i < input.length; i++) {
      buf[writeIdx] = input[i];
      writeIdx = (writeIdx + 1) % win;
      samplesSince++;
      if (samplesSince >= hop) {
        samplesSince = 0;
        // snapshot window in correct time order
        const frame = new Float32Array(win);
        const tail = win - writeIdx;
        frame.set(buf.subarray(writeIdx), 0);
        frame.set(buf.subarray(0, writeIdx), tail);
    
        const f = extractFeatures(frame, sr);
        f.dt = hop / sr;
        applyFeatures(f);
      }
    }
    };

