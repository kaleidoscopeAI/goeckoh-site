export default function App3D() {
const mountRef = useRef<HTMLDivElement>(null);
const loopRef = useRef<EmotionLoop | null>(null);
const nodeMeshesRef = useRef<THREE.Mesh[]>([]);
const vectorArrowsRef = useRef<THREE.ArrowHelper[][]>([]);
const constructLinesRef = useRef<THREE.Line[][]>([]);
