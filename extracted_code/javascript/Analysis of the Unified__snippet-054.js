// Return buffer to pool
const bufferIndex = buffers.findIndex(buf => buf.buffer === data.buffer);
if (bufferIndex !== -1) {
  availableBuffers.push(bufferIndex);
}
