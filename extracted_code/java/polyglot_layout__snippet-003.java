            for (int i = 0; i < dim; i++) v.set(i, v.get(i) / norm);

        }

        return v;

    }

    public static int hashCode(String str) {

        int hash = 0;

        for (int i = 0; i < str.length(); i++) {

            hash = (hash << 5) - hash + str.charAt(i);

        }

        return hash;

    }

}

class AGIOrchestrator {

    private KnowledgeDNA dna;

    private AGIMathematics math;

    private MemoryStore memory;

    private List<String> history;

    private double phi;

    private boolean conscious;

    private volatile boolean running;

    public AGIOrchestrator(MemoryStore memory) {

        this.dna = new KnowledgeDNA();

        this.math = new AGIMathematics();

        this.memory = memory;

        this.history = new ArrayList<>();

        this.phi = 0.0;

        this.conscious = false;

        this.running = true;

        Runtime.getRuntime().addShutdownHook(new Thread(() -> {

            System.out.println("\nInterrupt signal received. Shutting down gracefully...");

            stop();

        }));

    }

    public void step() {

        String text = "Artificial Intelligence evolves";

        List<Double> vec = Utils.embedText(text, 256);

        this.phi = this.math.integratedInformation(vec);


