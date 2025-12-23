// main thread feedback forwarded to worker
const { requestId, analysis } = data;
const req = outstandingRequests.get(requestId);
if (req && quantumEngine) {
  quantumEngine.integrateOllamaFeedback(analysis);
  outstandingRequests.delete(requestId);
}
