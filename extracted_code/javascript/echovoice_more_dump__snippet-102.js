  const novelty = len(sub(input, this.retrieveMemory(input))); // FAISS dist
  const arousal = this.avgC[1]; // From emotional
  return novelty * arousal;
