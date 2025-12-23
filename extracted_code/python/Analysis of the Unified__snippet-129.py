def __init__(self, id):
    self.id=id
    self.r=[random.uniform(-1,1) for _ in range(3)]
    self.E,self.S=random.random(),random.random()
    self.V,self.A=0,0
    self.q=[complex(random.random(),random.random()) for _ in range(2)]
def update(self,neighbors):
    stress=sum(np.linalg.norm(np.array(self.r)-np.array(n.r)) for n in neighbors)
    self.S=0.5*self.S+0.5*stress
    self.V=math.tanh(self.E-self.S)
    self.A=math.exp(-abs(self.E-self.S))
    self.r=[x+self.V*0.05 for x in self.r]
