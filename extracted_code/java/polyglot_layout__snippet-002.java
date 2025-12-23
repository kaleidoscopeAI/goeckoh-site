        for (double d : data) {

            double p = Math.abs(d) / sum;

            if (p > 0.0) res -= p * Math.log(p);

        }

        return res;

    }

    public double integratedInformation(List<Double> vec) {

        if (vec == null || vec.isEmpty()) return 0.0;

        int n = vec.size();

        int parts = Math.max(1, n / 2);

        double sysEnt = entropy(vec);

        double partEnt = 0.0;

        for (int i = 0; i < parts; i++) {

            List<Double> subset = new ArrayList<>();

            for (int j = i; j < n; j += parts) subset.add(vec.get(j));

            partEnt += entropy(subset);

        }

        partEnt /= parts;

        return Math.max(0.0, sysEnt - partEnt);

    }

}

class KnowledgeDNA {

    int generation;

    public KnowledgeDNA() {

        this.generation = 0;

    }

    public void replicate() {

        this.generation++;

    }

}

class MemoryStore {

    private Connection conn;

    private MemoryStore(Connection conn) {

        this.conn = conn;

    }

    public static MemoryStore create(String path) throws SQLException {

        Connection conn = DriverManager.getConnection("jdbc:sqlite:" + path);


