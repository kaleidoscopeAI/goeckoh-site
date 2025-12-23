function split_text(string $s): array { return preg_split('/\s+/', trim($s)) ?: []; }
function embed_text(string $text, int $dim=256): array {
