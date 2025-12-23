        this.dna.replicate();

        await this.memory.saveState(this.dna.generation, this.phi);

        this.history.unshift(Φ=${this.phi.toFixed(3)});

        if (this.history.length > 1000) this.history.pop();

    }

    async run() {

        while (this.running) {

            try {

                await this.step();

                console.log(Tick: gen=${this.dna.generation}, Φ=${this.phi.toFixed(3)}, conscious=${this.conscious});

                await new Promise((res) => setTimeout(res, 1000));

            } catch (e) {

                console.error('Runtime error:', e.message);

                this.running = false;

            }

        }

        console.log('AGI Orchestrator stopped gracefully.');

        await this.memory.close();

    }

}

(async () => {

    try {

        const memory = await MemoryStore.create('agi_js.db');

        const agi = new AGIOrchestrator(memory);

        await agi.run();

    } catch (e) {

        console.error('Error:', e.message);

        process.exit(1);

    }

})();

import sqlite3 from 'sqlite3';

import { open, Database, Statement } from 'sqlite';

class AGIMathematics {

    entropy(data: number[]): number {

        if (!Array.isArray(data) || data.length === 0) return 0.0;

        const sum = data.reduce((acc, d) => acc + Math.abs(d), 0);

        if (sum <= 0.0) return 0.0;

        return data.reduce((res, d) => {

