n = len(nodes)
W = np.zeros((n, n), dtype=float)
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        ck = cos_sim(nodes[i].K, nodes[j].K)
        cp = cos_sim(nodes[i].Psi, nodes[j].Psi)
        cd = cos_sim(nodes[i].D, nodes[j].D)
        w = (ck + cp + cd) / 3.0
        w *= math.exp(-cfg.beta * np.linalg.norm(nodes[j].U))
        # role bias: prefer learning from same-role or trusted
        role_bias = 1.0 if nodes[i].R == nodes[j].R else 0.5
        W[i, j] = w * role_bias
return W


