const cleaned = m[0].replace(/,\s*}/g, '}').replace(/,\s*]/g, ']');
try { return JSON.parse(cleaned); } catch { return null; }
