        if (!this.conscious && this.phi > 0.7) {

            this.conscious = true;

            System.out.printf("Consciousness threshold reached: Φ=%.3f%n", this.phi);

        }

        this.dna.replicate();

        this.memory.saveState(this.dna.generation, this.phi);

        this.history.add(0, String.format("Φ=%.3f", this.phi));

        if (this.history.size() > 1000) this.history.remove(this.history.size() - 1);

    }

    public void stop() {

        this.running = false;

    }

    public void run() {

        while (this.running) {

            try {

                step();

                System.out.printf("Tick: gen=%d, Φ=%.3f, conscious=%b%n", this.dna.generation, this.phi, this.conscious);

                Thread.sleep(1000);

            } catch (InterruptedException e) {

                Thread.currentThread().interrupt();

                stop();

            } catch (Exception e) {

                System.err.println("Runtime error: " + e.getMessage());

                stop();

            }

        }

        System.out.println("AGI Orchestrator stopped gracefully.");

        this.memory.close();

    }

}

public class FinalAgiJava {

    public static void main(String[] args) {

        try {

            MemoryStore memory = MemoryStore.create("agi_java.db");

            AGIOrchestrator agi = new AGIOrchestrator(memory);

            agi.run();

        } catch (Exception e) {

            System.err.println("Error: " + e.getMessage());

