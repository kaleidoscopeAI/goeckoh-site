// Analysis window & hop
this.win = 2048;
this.hop = Math.max(1, Math.floor(this.sr * 0.01)); // ~10ms
this.dt = this.hop / this.sr;

this.ring = new Float32Array(this.win);
this.writeIdx = 0;
this.samplesSince = 0;

// Precompute Hann window
this.hann = new Float32Array(this.win);
for (let i = 0; i < this.win; i++) {
  this.hann[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / (this.win - 1));
}

// For FFT-ish magnitude (we do a real DFT for K bins, optimized range)
this.K = this.win / 2;

// Downsampled spectrum bins for tilt regression (reduce CPU)
// We compute magnitudes for bins 2..K-1 but only accumulate in desired band.
}

