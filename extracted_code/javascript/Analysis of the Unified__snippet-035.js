if (this.processing || this.requestQueue.length === 0) return;

this.processing = true;
const { request, resolve, reject } = this.requestQueue.shift()!;

try {
  const response = await this.makeRequest('POST', '/api/generate', request);
  resolve(response);
} catch (error) {
  reject(error as Error);
} finally {
  this.processing = false;
  this.processQueue(); // Process next item
}
