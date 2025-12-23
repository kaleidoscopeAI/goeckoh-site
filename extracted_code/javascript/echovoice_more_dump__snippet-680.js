const [applyRange, setApplyRange] = useState("5,10");
const [seedRange, setSeedRange] = useState("42,1337");
const [status, setStatus] = useState<string>("Idle");
async function runSweep() {
