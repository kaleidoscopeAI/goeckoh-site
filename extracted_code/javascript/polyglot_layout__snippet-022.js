            const p = Math.abs(d) / sum;

            return p > 0.0 ? res - p * Math.log(p) : res;

        }, 0.0);

    }

    integratedInformation(vec: number[]): number {

        if (!Array.isArray(vec) || vec.length === 0) return 0.0;

        const n = vec.length;

        const parts = Math.max(1, Math.floor(n / 2));

        const sysEnt = this.entropy(vec);

        let partEnt = 0.0;

        for (let i = 0; i < parts; i++) {

            const subset: number[] = [];

            for (let j = i; j < n; j += parts) subset.push(vec[j]);

            partEnt += this.entropy(subset);

        }

        partEnt /= parts;

        return Math.max(0.0, sysEnt - partEnt);

    }

}

class KnowledgeDNA {

    generation: number;

    constructor() {

        this.generation = 0;

    }

    replicate(): void {

        this.generation++;

    }

}

class MemoryStore {

    db: Database;

    insertStmt: Statement | null = null;

    constructor(db: Database) {

        this.db = db;

    }

    static async create(path: string): Promise<MemoryStore> {

        const db = await open({ filename: path, driver: sqlite3.Database });

        await db.exec('CREATE TABLE IF NOT EXISTS dna (gen INTEGER PRIMARY KEY, phi REAL);');


