public synchronized void saveState(int gen, double phi) {
String sql = "INSERT OR REPLACE INTO dna (gen, phi) VALUES (?, ?)";
