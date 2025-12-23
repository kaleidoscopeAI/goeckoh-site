  await ws.accept()
  try:
       # gentle warm-up / keep-alive
       while True:
           buf = orch.avatar.frame_bytes() if hasattr(orch, "avatar") else None
           if buf is not None:
               await ws.send_bytes(buf)
           # ~10 FPS, OK for 18k*3 float32 (~216 KB/frame)
           await asyncio.sleep(0.1)
  except WebSocketDisconnect:
       return



 If your current script defines orch = Orchestrator() near the bottom, the endpoint will read from it directly as shown.


