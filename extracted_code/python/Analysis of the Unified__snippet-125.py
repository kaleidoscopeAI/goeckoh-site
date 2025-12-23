def __init__(self):
    self.roots = [[1, -1, 0, 0, 0, 0, 0, 0], [1, 0, -1, 0, 0, 0, 0, 0], [0.5]*8]  # Including average for superposition

def project_to_8d(self, vec3d):
    try:
        vec8d = vec3d + [0]*5
        norm = math.sqrt(sum(x**2 for x in vec8d))
        if norm == 0: raise ValueError("Zero norm")
        return [x / norm for x in vec8d]
    except:
        return [0.125]*8  # Correction to uniform

def reflect(self, vec8d, root):
    try:
        dot_vr = sum(v*r for v,r in zip(vec8d, root))
        dot_rr = sum(r**2 for r in root)
        if dot_rr == 0: raise ValueError("Zero dot")
        proj = [dot_vr / dot_rr * r for r in root]
        reflected = [v - 2*p for v,p in zip(vec8d, proj)]
        return reflected
    except:
        return vec8d  # Correction

def mirror_state(self, vec3d):
    try:
        vec8d = self.project_to_8d(vec3d)
        root = random.choice(self.roots)
        return self.reflect(vec8d, root)[:3]
    except:
        return [0,0,0]

