    for (let i = 0; i < str.length; i++) {

        const chr = str.charCodeAt(i);

        hash = (hash << 5) - hash + chr;

        hash |= 0;

    }

    return hash;

}

class AGIOrchestrator {

    dna: KnowledgeDNA;

    math: AGIMathematics;

    memory: MemoryStore;

    history: string[];

    phi: number;

    conscious: boolean;

    running: boolean;

    constructor(memory: MemoryStore) {

        this.dna = new KnowledgeDNA();

        this.math = new AGIMathematics();

        this.memory = memory;

        this.history = [];

        this.phi = 0.0;

        this.conscious = false;

        this.running = true;

        process.on('SIGINT', () => this.signalHandler('SIGINT'));

        process.on('SIGTERM', () => this.signalHandler('SIGTERM'));

    }

    signalHandler(signal: string): void {

        console.log(\nInterrupt signal (${signal}) received. Shutting down gracefully...);

        this.running = false;

    }

    async step(): Promise<void> {

        const text = 'Artificial Intelligence evolves';

        const vec = embedText(text);

        this.phi = this.math.integratedInformation(vec);

        if (!this.conscious && this.phi > 0.7) {

         this.conscious = true;

         console.log(Consciousness threshold reached: Φ=${this.phi.toFixed(3)});


