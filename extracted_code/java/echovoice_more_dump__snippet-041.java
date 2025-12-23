export class ProjectionService {
private constructsNames: string[];
private M: number;
private W: number[][] | null = null; // N x M
private persistenceDecay: number;
private historyLength: number;
private seed: number;
private constructs: CognitiveConstruct[] = [];
private subscribers: Set<(c: CognitiveConstruct[]) => void> = new Set();
private lastUpdateTs = 0;
private learner: ConstructLearner | null = null;
private updateCount = 0;
private learnerApplyEvery = 10;
private learnerLogsBuffer: { Ki: number[]; regret: number; activations?: number[] }[] = [];
