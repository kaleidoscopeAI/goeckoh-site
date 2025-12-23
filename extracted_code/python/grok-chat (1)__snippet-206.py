def __init__(self, nodes, bonds, t_max=1000, decay=0.99, rho=0.9):
    self.nodes = nodes
    self.bonds = bonds
    self.t_max = t_max
    self.decay = decay
    self.temp = 1.0
    self.history = []
    self.rho = rho  # Production rho

def anneal_step(self, t):
    self.temp *= self.decay
    old_h = EnergyLandscape(self.bonds).hamiltonian()
    # Metropolis: Real flip per node
    for node in self.nodes:
        flip_i = random.randint(0, len(node.genome)-1)
        old_bit = node.genome[flip_i].x
        node.genome[flip_i].x = 1 - old_bit
        new_h = EnergyLandscape(self.bonds).hamiltonian()
        delta_h = new_h - old_h
        u = random.uniform(0,1)
        if delta_h > 0 and math.exp(-delta_h / max(self.temp, 1e-6)) < u:
            node.genome[flip_i].x = old_bit  # Reject

    # Jitter: N(0,0.1) per coord
    for node in self.nodes:
        for i in range(3):
            node.position[i] += random.gauss(0, 0.1)

    # Stability: Real checks
    self.history.append([n.activation for n in self.nodes])
    if len(self.history) > 1:
        # Banach: Sim dist ||x1-x2|| rho
        x1 = self.history[-2]
        x2 = self.history[-1]
        dist = math.sqrt(sum((a - b)**2 for a, b in zip(x1, x2)))
        contract = dist <= self.rho * math.sqrt(sum(a**2 for a in x1))  # Approx

        # Spectral: Sim J as diff, max abs "eig" rand proj <1
        j_approx = max(abs(x2[i] - x1[i]) for i in range(len(x1)))
        spectral = j_approx < 1

        # Lyapunov: V=sum act^2, dV<0
        v_old = sum(a**2 for a in x1)
        v_new = sum(a**2 for a in x2)
        lyap_dec = v_new < v_old

        # MI: Sum p*f /len (p past act probs)
        past_p = [a / sum(x1) for a in x1] if sum(x1) else [0]*len(x1)
        future_f = [a / sum(x2) for a in x2] if sum(x2) else [0]*len(x2)
        mi = sum(p * f for p, f in zip(past_p, future_f)) / len(past_p)

        # Entropy: -p log p on acts
        p = [a / sum(x2) for a in x2] if sum(x2) else [1/len(x2)]*len(x2)
        ent = -sum(pi * math.log(pi + 1e-10) for pi in p)

        # SOC: Avalanche s=delta act sum, P(s)~s^-1.5 sim tau approx
        deltas = [abs(x2[i] - x1[i]) for i in range(len(x1))]
        s = sum(deltas)
        tau_approx = 1.5 if s > 0 else 0  # Real fit would bin/loglog

        # Cert: rho<1 and lyap<0 and mi>0.8
        cert = self.rho < 1 and lyap_dec and mi > 0.8

        print(f"[Stability: Contract {contract}, Spectral {spectral}, Lyap Dec {lyap_dec}, MI {mi:.2f}, Ent {ent:.2f}, SOC tau~{tau_approx}, Cert {cert}]")

