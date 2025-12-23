import matplotlib.pyplot as plt

class DynamicVisualization:
    def __init__(self):
        self.fig, self.axs = plt.subplots(3, 1, figsize=(8, 12))

    def update(self, epochs, population, maturity, energy):
        """Update real-time graphs."""
        self.axs[0].clear()
        self.axs[0].plot(epochs, population, label="Population")
        self.axs[0].set_title("Population Growth")
        self.axs[0].legend()

        self.axs[1].clear()
        self.axs[1].plot(epochs, maturity, label="Average Maturity", color="orange")
        self.axs[1].set_title("Maturity Over Time")
        self.axs[1].legend()

        self.axs[2].clear()
        self.axs[2].plot(epochs, energy, label="Energy Levels", color="green")
        self.axs[2].set_title("Energy Usage")
        self.axs[2].legend()

        plt.pause(0.1)

