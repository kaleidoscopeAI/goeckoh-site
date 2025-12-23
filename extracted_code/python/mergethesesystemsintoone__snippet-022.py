async def think(payload: Dict = Body(...)):
    text = payload.get("text")
    return await brain.think(text)

