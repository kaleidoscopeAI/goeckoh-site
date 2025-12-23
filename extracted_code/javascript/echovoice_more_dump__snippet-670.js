export default function Dashboard() {
const [service] = useState(() => new ProjectionService({ seed: 1337, persistenceDecay: 0.94 }));
const { e, setE, resetNeutral } = useEmotionalActuation();
const [selectedWId, setSelectedWId] = useState<string | null>(null);
const [wList, setWList] = useState<any[]>([]);
const chartRef = useRef<HTMLCanvasElement | null>(null);
const [timeseries, setTimeseries] = useState<{ts:number, vals:number[]}[]>([]);
