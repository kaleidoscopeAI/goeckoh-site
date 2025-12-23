class CrystallineLattice:
    # ...

    def emotional_ode_step(self, B: np.ndarray, input_stimulus: Optional[np.ndarray], dt: float) -> None:
        E = self.emotions

        drive = self.alpha_drive * (input_stimulus if input_stimulus is not None else np.zeros_like(E))
        # ... (rest same)

    def step(self, input_stimulus: Optional[np.ndarray] = None, dt: float = 0.05) -> Dict[str, Any]:
        B = self.bond_weights()
        self.emotional_ode_step(B, input_stimulus, dt)  # Now semantic-driven

        # ...
