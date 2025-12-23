H=max(0.0,min(1.0,H_bits)); S=max(0.0,min(1.0,S_field)); L=max(0.0,min(1.0,latency)); F=max(0.0,min(1.0,fitness))
def a_fn(t): return 0.25 + 0.5*(1.0-H)*(1.0-S)
def m_fn(t): return 2.0 + 10.0*S
def rho_fn(t): return 0.2 + 3.0*(1.0-L)
def fc_fn(t): return fmin + fdelta*F
return {"a":a_fn,"m":m_fn,"rho":rho_fn,"fc":fc_fn}

