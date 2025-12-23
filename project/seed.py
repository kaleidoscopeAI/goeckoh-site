import math, random, asyncio, json, pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from .config import Config
from .ethics import EthicalGovernor
from .crystal_memory import CrystalMemory
from .gears import GearNode, GearFabric, Message
from .web_crawler import WebCrawler
from .hid_controller import HIDController

@dataclass
class SeedState:
    energy: float
    complexity: int
    dna: List[float]
    traits: Dict[str, float]
    fitness: float
    anneal_temp: float

class Seed:
    def __init__(self, cfg: Config):
        self.cfg=cfg
        self.dna=[random.uniform(0.05,0.95) for _ in range(cfg.dna_size)]
        self.max_energy=cfg.max_energy; self.energy=cfg.max_energy*0.8; self.complexity=0
        self.fitness_trace=[]
        self.traits={'learning_rate':self._scale(self.dna[0],0.02,0.3),
                     'efficiency':self._scale(self.dna[1],0.5,2.0),
                     'creativity':self._scale(self.dna[2],0.1,0.8),
                     'ethics_weight':self._scale(self.dna[3],0.3,0.9),
                     'replication_prob':self._scale(self.dna[4],0.1,0.7)}
        self.ethics=EthicalGovernor(); self.crystal=CrystalMemory(sr=cfg.audio_sr, base_freq=cfg.audio_base_freq)
        self.crawler=WebCrawler(cfg.allowed_domains) if cfg.crawler_enabled else None
        self.hid=HIDController(safety_mode=True) if cfg.hid_enabled else None
        self.core=GearNode("Core", self._on_core); self.reflect=GearNode("Reflection", self._on_reflect); self.emotion=GearNode("Emotion", self._on_emotion)
        for g in [self.core,self.reflect,self.emotion]: GearFabric.register(g)
        self.core.connect(self.emotion); self.emotion.connect(self.reflect)
        for g in [self.core,self.reflect,self.emotion]: g.start()
        self._anneal_k=0

    def _scale(self,x,lo,hi):
        s=1/(1+math.exp(-10*(x-0.5))); return lo+s*(hi-lo)

    def update_energy(self, success, dt=1.0):
        basal=0.1*self.traits['efficiency']
        learn=0.5*success*self.traits['learning_rate']
        dE=(-basal+learn)*dt; self.energy=max(0.0,min(self.max_energy,self.energy+dE)); return dE

    def mutate(self,sigma=0.02):
        fitness=sum(self.fitness_trace[-10:])/10.0 if len(self.fitness_trace)>=10 else 0.5
        s=sigma*(1.0-fitness); 
        return [min(0.999,max(0.001,g+random.gauss(0,s))) for g in self.dna]

    def can_replicate(self):
        ratio=self.energy/self.max_energy; threshold=0.4+0.3*self.traits['replication_prob']; return ratio>threshold

    def replicate(self):
        if self.energy<self.cfg.replication_energy_cost: return None
        env={'resource_capacity':10.0,'population':max(1,len(GearFabric.nodes)),'risk_threshold':self.cfg.ethics_risk_threshold}
        ok=self.ethics.evaluate_replication({'efficiency':self.traits['efficiency'],'replication_prob':self.traits['replication_prob'],'ethics_weight':self.traits['ethics_weight']},env)
        if not ok: return None
        self.energy-=self.cfg.replication_energy_cost
        self.dna=self.mutate(); return True

    def learn(self, datum):
        sc=random.random()
        if sc>0.5: self.complexity+=1
        self.update_energy(sc); self.fitness_trace.append(sc); return sc

    def _on_core(self, msg: Message):
        if isinstance(msg.content, dict) and msg.content.get('tick'):
            self._anneal_k+=1
            vecs=[[*self.dna],[self.traits['learning_rate'],self.traits['efficiency'],self.traits['creativity'],self.traits['ethics_weight'],self.traits['replication_prob']]]
            self.crystal.anneal(vecs, steps=8, t0=1.0, t1=0.1)
            if self._anneal_k % 20 == 0:
                self.crystal.synthesize_wav('/mnt/data/seed_memory.wav', duration_s=2.0)
        return None
    def _on_emotion(self, msg: Message):
        self.traits['replication_prob']=max(0.1,min(0.7,self.traits['replication_prob']+(random.random()-0.5)*0.02)); return {'mood':'ok'}
    def _on_reflect(self, msg: Message):
        return {'reflection':f"Energy={self.energy:.2f} Complexity={self.complexity} Fitness={sum(self.fitness_trace[-10:])/max(1,len(self.fitness_trace[-10:])):.2f}"}
    def snapshot(self):
        temp=self.cfg.anneal_temp_start+(self.cfg.anneal_temp_end-self.cfg.anneal_temp_start)*(len(self.fitness_trace)%self.cfg.anneal_steps)/max(1,self.cfg.anneal_steps-1)
        return {'name':self.cfg.seed_name,'energy':round(self.energy,3),'complexity':self.complexity,'dna':[round(x,3) for x in self.dna],
                'traits':{k:round(v,3) for k,v in self.traits.items()},'fitness':round(sum(self.fitness_trace[-10:])/max(1,len(self.fitness_trace[-10:])),3),'anneal_temp':round(temp,3),'gears':__import__('organic_ai.gears',fromlist=['GearFabric']).GearFabric.snapshot()}
    async def run(self):
        if self.crawler: self.crawler.seed([f"https://{d}/" for d in self.cfg.allowed_domains])
        t=0
        while True:
            self.core.send(None, {'tick': True})
            if self.crawler and t%8==0:
                for doc in self.crawler.step(limit=2): self.learn({'concept':doc['title'],'text':doc['text']})
            if t%25==0 and self.can_replicate(): self.replicate()
            pathlib.Path('/mnt/data/seed_state.json').write_text(json.dumps(self.snapshot()))
            await asyncio.sleep(0.2); t+=1
