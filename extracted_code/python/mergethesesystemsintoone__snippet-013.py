def synth_signal(seconds: float, sr: int, a_fn, m_fn, rho_fn, fc_fn, alpha: float = 0.8, beta: float = 0.4)->List[float]:
    n=int(seconds*sr); out=[]
    for i in range(n):
        t=i/sr; a=a_fn(t); m=m_fn(t); rho=rho_fn(t); fc=max(5.0, fc_fn(t))
        y = a*(1.0+beta*math.sin(2*math.pi*m*t))*math.sin(2*math.pi*fc*t + alpha*math.sin(2*math.pi*rho*t))
        out.append(y)
    return out
def default_maps(H_bits:float, S_field:float, latency:float, fitness:float, fmin:float=110.0, fdelta:float=440.0):
    H=max(0.0,min(1.0,H_bits)); S=max(0.0,min(1.0,S_field)); L=max(0.0,min(1.0,latency)); F=max(0.0,min(1.0,fitness))
    def a_fn(t): return 0.25 + 0.5*(1.0-H)*(1.0-S)
    def m_fn(t): return 2.0 + 10.0*S
    def rho_fn(t): return 0.2 + 3.0*(1.0-L)
    def fc_fn(t): return fmin + fdelta*F
    return {"a":a_fn,"m":m_fn,"rho":rho_fn,"fc":fc_fn}

