from scipy.integrate import odeint  # Real orbits

def compute_orbit(states, t):
    # Simple 2-body ode from tools
    r = np.linalg.norm(states[:3] - states[3:6])
    acc1 = - (states[3:6] - states[:3]) / r**3
    acc2 = - acc1
    return np.concatenate([states[3:6], acc1, states[9:12], acc2])  # dpos/dt = vel, dvel/dt = acc

