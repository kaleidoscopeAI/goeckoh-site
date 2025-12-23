for (let i = 0; i < ch.length; i++) {
  const x = ch[i];
  this.ring[this.writeIdx] = x;
  this.writeIdx = (this.writeIdx + 1) % this.win;

  this.samplesSince++;
  if (this.samplesSince >= this.hop) {
    this.samplesSince = 0;
    const frame = this._snapshotFrame();
    const feat = this._extract(frame);
    feat.dt = this.dt;
    this.port.postMessage(feat);
  }
}
return true;
}

