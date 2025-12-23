// main thread returned buffer; push index back based on buffer reference
// buffer arrives as ArrayBuffer — find matching index by comparing byteLength
const buff = data.buffer as ArrayBuffer;
const matchIndex = buffers.findIndex(b => b.buffer.byteLength === buff.byteLength && b.buffer !== buff);
// best-effort: if no exact match, push the first free index
if (matchIndex !== -1) availableBuffers.push(matchIndex);
else availableBuffers.push(0);
