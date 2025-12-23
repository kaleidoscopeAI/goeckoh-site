def sha_to_u64(s: str, salt: str="")->int:
    import hashlib
    h=hashlib.sha256((salt+s).encode("utf-8","ignore")).digest()
    return int.from_bytes(h[:8],"little")
def u_hash(x:int, a:int=_A, b:int=_B, p:int=_P, D:int=512)->int:
    return ((a*x+b)%p)%D
def sign_hash(x:int)->int:
    return 1 if (x ^ (x>>1) ^ (x>>2)) & 1 else -1
def ngrams(s:str, n_min=1, n_max=4)->List[str]:
    s = "".join(ch for ch in s.lower())
    grams = []
    for n in range(n_min, n_max+1):
        for i in range(len(s)-n+1):
            grams.append(s[i:i+n])
    return grams
def embed_text(text:str, D:int=512)->np.ndarray:
    v=np.zeros(D,dtype=np.float64)
    for g in ngrams(text,1,4):
        x=sha_to_u64(g)
        d=u_hash(x,D=D)
        s=sign_hash(sha_to_u64(g,"sign"))
        v[d]+=s
    nrm=np.linalg.norm(v)+1e-9
    return v/nrm

