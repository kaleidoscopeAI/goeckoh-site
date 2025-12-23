const merged = { ...this.localE, ...newE } as EVector;
const sanitized = this._sanitizeAndClampVector(merged);
