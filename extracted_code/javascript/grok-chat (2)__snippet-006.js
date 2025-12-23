export default function JacksonCompanion() {
  const [nodes, setNodes] = useState([]);
  const [edges, setEdges] = useState([]);
  const [running, setRunning] = useState(false);
  const [angle, setAngle] = useState(0);
  const canvasRef = useRef(null);
  const semRef = useRef(new SemanticEngine());
 
  const [phrase, setPhrase] = useState('hello');
  const [input, setInput] = useState('');
  const [attempts, setAttempts] = useState([]);
  const [result, setResult] = useState(null);
 
  const ag = useRef({t:0,will:0,eta:.05,lam:1,gcl:.5,phi:0,pain:0,risk:0,rate:.5,life:0,DA:.5,Ser:.5,NE:.3,hist:[],pSint:.5,pN:0,evt:null});
  const [ds, setDs] = useState({t:0,phi:0,gcl:.5,pain:0,life:0,risk:0,rate:.5,total:0,corr:0,DA:.5,Ser:.5,NE:.3,thought:'Awakening...',calm:'',concepts:[],energy:0});
  const [log, setLog] = useState([]);
  const init = useCallback((n=50)=>{
    const ns = Array.from({length:n},(*,i)=>({id:i,s:randU32(),x:randPos(),e:0}));
    const es = [];
    for(let u=0;u<n;u++){ const nb=new Set(); while(nb.size<Math.min(4,n-1)){const v=Math.floor(Math.random()*n);if(v!==u)nb.add(v);} nb.forEach(v=>{if(!es.some(e=>(e[0]===u&&e[1]===v)||(e[0]===v&&e[1]===u)))es.push([u,v]);}); }
