def cos_sim(a:np.ndarray, b:np.ndarray, eps:float=1e-9)->float:
    return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+eps))
def knn_idx(E:np.ndarray, i:int, k:int=8)->List[int]:
    x=E[i]; sims=(E@x)/(np.linalg.norm(E,axis=1)*(np.linalg.norm(x)+1e-9)+1e-12)
    order=np.argsort(-sims); return [j for j in order if j!=i][:k]
def mc_var(E:np.ndarray, i:int, k:int, sigma:float, M:int=6, rng=None)->float:
    if rng is None: rng=np.random.RandomState(7)
    idx=knn_idx(E,i,k=max(1,min(k,E.shape[0]-1))); vals=[]; D=E.shape[1]
    for _ in range(M):
        ei=E[i]+sigma*rng.normal(0.0,1.0,size=D); ei/= (np.linalg.norm(ei)+1e-9)
        sims=[]
        for j in idx:
            ej=E[j]+sigma*rng.normal(0.0,1.0,size=D); ej/= (np.linalg.norm(ej)+1e-9)
            sims.append(cos_sim(ei,ej))
        vals.append(max(sims) if sims else 0.0)
    return float(np.var(vals))
def stability(var_sigma:float)->float: return 1.0/(1.0+var_sigma)
def anneal_sigma(sigma0:float, gamma:float, step:int, sigma_min:float)->float:
    return max(sigma0*(gamma**step), sigma_min)
def expected_cos_noise(ei,ej,sigma,M=4)->float:
    rng=np.random.RandomState(11); sims=[]
    for _ in range(M):
        ein=ei+sigma*rng.normal(0.0,1.0,size=ei.shape); ein/= (np.linalg.norm(ein)+1e-9)
        ejn=ej+sigma*rng.normal(0.0,1.0,size=ej.shape); ejn/= (np.linalg.norm(ejn)+1e-9)
        sims.append(float(np.dot(ein,ejn)))
    return float(np.mean(sims))
def ring_edges(N:int,k:int=6)->np.ndarray:
    edges=set()
    for i in range(N):
        edges.add(tuple(sorted((i,(i+1)%N))))
        for d in range(1,k//2+1): edges.add(tuple(sorted((i,(i+d)%N))))
    if not edges: return np.zeros((0,2),dtype=np.int32)
    return np.array(sorted(list(edges)),dtype=np.int32)
def energetics(E:np.ndarray, S:np.ndarray, edges:np.ndarray, sigma:float)->dict:
    if len(edges)==0:
        return {"H_bits": float(np.mean(1.0-S) if E.shape[0] else 0.0), "S_field":0.0, "L":0.0}
    w=np.zeros(len(edges)); 
    for k,(i,j) in enumerate(edges): w[k]=expected_cos_noise(E[i],E[j],sigma)
    tau=1.0-w
    H_bits=float(np.mean(1.0-S) if E.shape[0] else 0.0)
    S_field=float(np.mean(tau)); L=float(np.sum(tau*tau))
    return {"H_bits":H_bits,"S_field":S_field,"L":L}

