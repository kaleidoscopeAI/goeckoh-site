export default function Constructs3D({ service, cfg, width = window.innerWidth, height = window.innerHeight, zDepth = 0 }: Props) {
const mountRef = useRef<HTMLDivElement | null>(null);
const sceneRef = useRef<THREE.Scene | null>(null);
const spritesRef = useRef<Record<string, THREE.Sprite>>({});
const ribbonsRef = useRef<Record<string, THREE.Line>>({});
const localServiceRef = useRef<ProjectionService | null>(null);
