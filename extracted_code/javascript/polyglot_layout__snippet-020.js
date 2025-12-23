    const norm = Math.sqrt(v.reduce((acc, x) => acc + x * x, 0));

    if (norm > 0) for (let i = 0; i < dim; i++) v[i] /= norm;

    return v;

}

function hashCode(str) {

    let hash = 0;

    for (let i = 0; i < str.length; i++) {

        const chr = str.charCodeAt(i);

        hash = (hash << 5) - hash + chr;

        hash |= 0;

    }

    return hash;

}

class AGIOrchestrator {

    constructor(memory) {

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

    signalHandler(signal) {

        console.log(\nInterrupt signal (${signal}) received. Shutting down gracefully...);

        this.running = false;

    }

    async step() {

        const text = 'Artificial Intelligence evolves';

        const vec = embedText(text);

        this.phi = this.math.integratedInformation(vec);

        if (!this.conscious && this.phi > 0.7) {

            this.conscious = true;

            console.log(Consciousness threshold reached: Φ=${this.phi.toFixed(3)});

        }


