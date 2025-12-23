const memory = await MemoryStore.create('agi_ts.db');
const agi = new AGIOrchestrator(memory);
