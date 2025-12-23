export default function CognitiveProjection({ nodesProvider, width = 640, height = 480, cfg, onInspect, showGrid = true }: Props) {
const { constructs, update, service } = useCognitiveProjection(nodesProvider, cfg);
