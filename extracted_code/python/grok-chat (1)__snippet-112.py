def therapy_entropy(text):
    if not text:
        return 0
    codes = [ord(c) for c in text]
    hist = [codes.count(i) for i in set(codes)]
    total = sum(hist)
    p = [h / total for h in hist]
    return -sum(pi * math.log(pi + 1e-10) for pi in p) if p else 0

# Optimized Life Eq: Params from sim [1.0,1.0,1.0]
def life_equation(states, env, past, future, n_copies, dt, lambda1=1.0, lambda2=1.0, lambda3=1.0):
    ds_int = (therapy_entropy(states) - therapy_entropy(past)) / dt if past else 0
    ds_env = random.gauss(0, 0.1)
    term1 = lambda1 * (ds_int - ds_env)

    mi = sum(p * f for p, f in zip(past or [0], future or [0])) / max(len(past), 1)
    h_x = therapy_entropy(states)
    term2 = lambda2 * (mi / (h_x + 1e-10))

    dn = n_copies / dt
    term3 = lambda3 * (dn / (len(states) + 1e-10))

    l = term1 + term2 + term3
    return (math.tanh(l) + 1) / 2

