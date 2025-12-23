if (!canvasRef.current || !imageData) return;
const ctx = canvasRef.current.getContext("2d");
if (!ctx) return;

const imgData = new ImageData(imageData, width, height);
ctx.putImageData(imgData, 0, 0);
