    for (let i = 0; i < this.win; i++) {
      this.hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (this.win - 1));
    }
    
    // For FFT-ish magnitude (we do a real DFT for K bins, optimized range)
    this.K = this.win / 2;
    
    // Downsampled spectrum bins for tilt regression (reduce CPU)
    // We compute magnitudes for bins 2..K-1 but only accumulate in desired band.
    }

