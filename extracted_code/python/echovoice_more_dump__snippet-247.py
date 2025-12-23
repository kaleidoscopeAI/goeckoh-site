def speculate(snapshot: dict):
    if 'orbit' in snapshot:
        sol = odeint(compute_orbit, snapshot['states'], np.linspace(0, 10, 100))  # Real integrate orbits
        return {"orbits": sol.tolist()}
    # ... Previous

