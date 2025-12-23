class MemoryStore {
private ?\SQLite3 $db = null;
public function __construct(string $path) {
