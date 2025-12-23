class GradientFlow:
    # ...
    def step(self, dt: float) -> None:
        # Calculates the true gradient of the energy function.
        grads = self.ham.analytic_gradient(self.state)
        # Updates the state's vectors by taking a small step in the opposite direction of the gradient.
        for n, g in grads.items():
            arr = np.asarray(self.state.x[n]).astype(float)
            self.state.x[n] = (arr - dt * self.lr * g).astype(float)

