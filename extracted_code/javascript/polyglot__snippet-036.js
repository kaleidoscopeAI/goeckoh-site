const memory = await MemoryStore.create('agi_js.db');
const agi = new AGIOrchestrator(memory);
