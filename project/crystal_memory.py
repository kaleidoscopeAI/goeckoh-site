import math, wave, struct, random
class CrystalMemory:
    def __init__(self, sr=22050, base_freq=220.0):
        self.sr=sr; self.base=base_freq; self.facets=[]
    def _dist(self,a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b))/max(1,len(a)))
    def anneal(self, vectors, steps, t0, t1):
        if not vectors: return []
        facets=[{"vector":vectors[0][:],"weight":1.0}]
        for s in range(1,steps):
            temp=t0+(t1-t0)*s/max(1,steps-1)
            v=random.choice(vectors)
            bi=min(range(len(facets)), key=lambda i:self._dist(facets[i]["vector"],v))
            d=self._dist(facets[bi]["vector"],v)+1e-6
            alpha=math.exp(-d/max(0.001,temp))
            facets[bi]["vector"]=[(1-alpha)*a+alpha*b for a,b in zip(facets[bi]["vector"],v)]
            facets[bi]["weight"]=min(5.0,facets[bi]["weight"]+alpha*0.1)
            if random.random()<max(0.01,temp*0.02): facets.append({"vector":v[:],"weight":0.5})
        self.facets=facets; return facets
    def synthesize_wav(self, path, duration_s=2.0):
        n=int(self.sr*duration_s); buf=bytearray()
        for i in range(n):
            t=i/self.sr; s=0.0
            for f in self.facets:
                mean=sum(f["vector"])/max(1,len(f["vector"])); freq=self.base*(1.0+mean)
                s+=f["weight"]*math.sin(2*math.pi*freq*t)
            s*=0.2; s=max(-1.0,min(1.0,s)); buf+=struct.pack('<h', int(s*32767))
        with wave.open(path,'wb') as wf:
            wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(self.sr); wf.writeframes(bytes(buf))
        return path
