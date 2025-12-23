export class EmotionalActuationService {
private app: any | null = null;
private db: Firestore | null = null;
private docRef: DocumentReference<DocumentData>;
private historyCollection: string;
private emotions: EmotionName[];
private normalized: ActuationDoc["normalized"];
private localE: EVector;
private subscribers: Set<(v: EVector) => void> = new Set();
private unsubscribeSnapshot: (() => void) | null = null;
private lastWriteTs = 0;
private smoothingMs: number;
