private W: number[][] | null = null; // [N x M]
private persistenceDecay: number;
private historyLength: number;
private seed: number;
private subscribers: Set<Subscriber> = new Set();
private constructs: CognitiveConstruct[] = [];
private lastUpdateTs = 0;
