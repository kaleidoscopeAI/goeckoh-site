class AGISystem:
    def __init__(self, n_nodes: int = 32, n_apparatus: int = 24):
        self.hw = SimulatedHardware(adc_channels=24)
        self.rel = RelationalMatrix(n_nodes, n_apparatus)
        self.thought = ThoughtEngines(n_nodes)
        self.emotion = EmotionalChemistry()
        self.memory = MemorySystem(embedding_dim=128, capacity=10000)
        self.planner = Planner(self.hw, self.rel)
        self.t = 0.0
        self.dt = 0.1
        self.history = []
        self.lock = Lock()
        # seed semantic memory
        self.memory.store_semantic("agent_identity", "A private, self-contained cognitive assistant for execution and opportunity synthesis.")
        self.memory.store_episode("system initialized")

    def step(self, user_input: Optional[str] = None) -> Dict[str, Any]:
        with self.lock:
            self.hw.poll_sensors()
            n = self.thought.n
            sensor_vec = np.concatenate([self.hw.adc, np.array([self.hw.temp_C, self.hw.freq_GHz])])
            inputs = np.zeros(n)
            ssum = float(np.sum(sensor_vec))
            for i in range(n):
                inputs[i] = float(np.tanh(ssum * 0.0005 + random.random() * 0.01))
            if user_input:
                self.memory.store_episode(user_input)
                emb = self.memory.embed(user_input)
                bias = float(np.tanh(np.mean(emb))) * 0.5
                inputs += bias
            self.thought.step(self.rel, inputs, dt=self.dt)
            reward = float(np.clip(np.mean(inputs), -1, 1))
            mood = float(np.tanh(np.mean(self.thought.b)))
            arousal = float(np.abs(np.std(self.thought.h)))
            self.emotion.step(reward, mood, arousal, dt=self.dt)
            most_active_node = int(np.argmax(np.abs(self.thought.kappa)))
            apparatus_idx = int(abs(int((np.sum(self.hw.adc) * 100) % self.rel.n_apparatus)))
            self.rel.update_hebbian(most_active_node, apparatus_idx, lr=1e-3)
            action_name = self.planner.select_and_execute(self.thought)
            self.rel.normalize_rows()
            conn_metrics = self.relational_consciousness_metrics()
            log_item = {
                "t": self.t,
                "action": action_name,
                "hw": self.hw.as_status(),
                "emotion": {"DA": self.emotion.DA, "Ser": self.emotion.Ser, "NE": self.emotion.NE},
                "thought_summary": {
                    "b_mean": float(np.mean(self.thought.b)),
                    "h_mean": float(np.mean(self.thought.h)),
                    "kappa_mean": float(np.mean(self.thought.kappa)),
                    "mu_mean": float(np.mean(self.thought.mu)),
                },
                "consciousness": conn_metrics
            }
            self.history.append(log_item)
            self.t += self.dt
            return log_item

    def relational_consciousness_metrics(self) -> Dict[str, float]:
        diag = np.array([abs(self.rel.R[i, i % self.rel.n_apparatus]) for i in range(min(self.rel.n_system, self.rel.n_apparatus))])
        coherence = float(np.mean(diag))
        awareness = float(np.clip(self.emotion.DA * (1.0 + np.tanh(np.mean(self.thought.b))), 0.0, 1.0))
        activities = np.concatenate([self.thought.b, self.thought.h, self.thought.kappa, self.thought.mu])
        integrated_info = float(np.var(activities))
        return {"coherence": coherence, "awareness": awareness, "phi_proxy": integrated_info}

    def respond(self, user_input: str) -> str:
        log = self.step(user_input)
        candidates = self.memory.retrieve(user_input, top_k=3)
        reply_parts = []
        if candidates:
            reply_parts.append("I recall: " + "; ".join([c for _, _, c in candidates[:2]]))
        if log["consciousness"]["awareness"] > 0.6:
            reply_parts.append("I am engaged and reflecting on that.")
        elif log["consciousness"]["phi_proxy"] > 0.08:
            reply_parts.append("This seems important; I'll think further.")
        else:
            reply_parts.append("Noted and stored.")
        da = log["emotion"]["DA"]
        mood = "positive" if da > 0.55 else "neutral" if da > 0.45 else "cautious"
        reply_parts.append(f"My mood is {mood}. Action taken: {log['action']}.")
        return " ".join(reply_parts)-e 


