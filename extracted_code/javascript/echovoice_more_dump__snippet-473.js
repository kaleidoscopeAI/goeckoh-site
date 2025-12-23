export default function AppCinematic3D() {
const mountRef = useRef<HTMLDivElement>(null);
const nodeParticlesRef = useRef<THREE.Points[][]>([]);
const nodeMeshesRef = useRef<THREE.Mesh[]>([]);
const constructLinesRef = useRef<THREE.Line[][]>([]);
const loopRef = useRef<EmotionLoop | null>(null);
