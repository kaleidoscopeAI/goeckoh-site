def life_equation(states, env, past, future, n_copies, dt):
    ds_int = (entropy(states) - entropy(past)) / dt if past else 0
    ds_env = random.gauss(0, 0.1)
    term1 = ds_int - ds_env

    mi = mutual_info(past, future)
    h_x = entropy(states)
    term2 = mi / (h_x + 1e-10) if h_x else 0

    dn = n_copies / dt
    term3 = dn / (len(states) + 1e-10)

    l = 0.4 * term1 + 0.3 * term2 + 0.3 * term3
    return (math.tanh(l) + 1) / 2  # Normalize 0-1

def entropy(values):
    if not values:
        return 0
    hist = [math.exp(-v**2) for v in values]  # Approx
    total = sum(hist)
    p = [h / total for h in hist]
    return -sum(pi * math.log(pi + 1e-10) for pi in p)

def mutual_info(past, future):
    return sum(p * f for p, f in zip(past or [0], future or [0])) / max(len(past), 1)

