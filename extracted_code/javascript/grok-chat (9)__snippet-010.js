     15    const [isConnected, setIsConnected] = useState<boolean>(false);
     16 +  const [isSseConnected, setIsSseConnected] = useState<boolean>(false);
     17    const [initialPrompt, setInitialPrompt] = useState<string>('The consc
         iousness of an AI visualizing its own neural network as a blooming flow
         er in space.');
        ⋮
    160
    161 +  // Subscribe to speech mirror SSE stream (backend /mirror/stream on p
         ort 8080)
    162 +  useEffect(() => {
    163 +    const proto = window.location.protocol === 'https:' ? 'https:' : 'h
         ttp:';
    164 +    const host = window.location.hostname || 'localhost';
    165 +    const url = `${proto}//${host}:8080/mirror/stream`;
    166 +    let lastTs = 0;
    167 +    let closed = false;
    168 +    let es: EventSource | null = null;
    169 +
    170 +    try {
    171 +      es = new EventSource(url);
    172 +      es.onopen = () => setIsSseConnected(true);
    173 +      es.onmessage = async (evt) => {
    174 +        if (closed) return;
    175 +        try {
    176 +          const data = JSON.parse(evt.data);
    177 +          const ts = data.ts || Date.now();
    178 +          if (ts === lastTs) return;
    179 +          lastTs = ts;
    180 +          const text = data.corrected || data.transcript || '';
    181 +          if (!text) return;
    182 +          const newTargets = await enhancedService.processConversationR
         esponse(text, false);
    183 +          setTargets(newTargets);
    184 +          setContext(enhancedService.getCurrentContext());
    185 +        } catch (err) {
    186 +          console.warn('SSE parse/process error', err);
    187 +        }
    188 +      };
    189 +      es.onerror = () => {
    190 +        setIsSseConnected(false);
    191 +        es && es.close();
    192 +      };
    193 +    } catch (err) {
    194 +      console.warn('SSE init failed', err);
    195 +    }
    196 +
    197 +    return () => {
    198 +      closed = true;
    199 +      if (es) es.close();
    200 +    };
    201 +  }, [enhancedService]);
    202 +
    203    return (
        ⋮
    216              </p>
    217 +            <div className="mt-2 text-xs text-gray-500">
    218 +              Mirror stream: {isSseConnected ? <span className="text-em
         erald-300">live</span> : 'disconnected'}
    219 +            </div>
    220          </header>

