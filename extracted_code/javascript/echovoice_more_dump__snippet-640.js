const service = useMemo(() => new ProjectionService(cfg), [] as any);
const [constructs, setConstructs] = useState<CognitiveConstruct[]>(service.getConstructs());
