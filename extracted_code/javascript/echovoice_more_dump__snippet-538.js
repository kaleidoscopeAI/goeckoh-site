export function useEmotionalActuation(firebaseConfig?: any): HookReturn {
const svcRef = useRef<EmotionalActuationService | null>(null);
const [eVec, setEVec] = useState<EVector>(() => ({} as any));
