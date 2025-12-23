const PORT = process.env.PERSISTENCE_PORT ? Number(process.env.PERSISTENCE_PORT) : 4201;
app.listen(PORT, () => console.log(`Persistence server listening on ${PORT}`));
