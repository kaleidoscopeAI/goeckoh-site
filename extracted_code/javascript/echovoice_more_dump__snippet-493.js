export default function AppCinematicNetwork3D() {
const mountRef = useRef<HTMLDivElement>(null);
const nodeMeshesRef = useRef<THREE.Mesh[]>([]);
const constructLinesRef = useRef<THREE.Line[][]>([]);
const nodeParticlesRef = useRef<THREE.Points[][]>([]);
const networkFlowsRef = useRef<THREE.Line[]>([]);
const loopRef = useRef<EmotionLoop | null>(null);
