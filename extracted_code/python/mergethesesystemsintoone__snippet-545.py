def __init__(self) -> None:
    self.processed_hashes: set[str] = set()
    self.model = SentenceTransformer('all-MiniLM-L6-v2')

def process_web_content(self, content: Dict[str, str]) -> Optional[List[Dict[str, str]]]:
    content_hash = hash(json.dumps(content))
    if content_hash in self.processed_hashes:
        return None
    concepts: List[Dict[str, str]] = []
    base_meta = {
        "source": content.get("url", ""),
        "timestamp": time.strftime("%Y-%m-%d_%H-%M-%S"),
    }
    for concept in content.get("concepts", [])[:5]:
        if len(concept) < 4:
            continue
        item = {
            "type": "concept",
            "content": f"{concept}: {content.get('title', 'Unknown')}",
            "complexity": min(len(concept.split()) * 0.1, 1.0),
            **base_meta,
        }
        concepts.append(item)
    sentences = content.get("content", "").split(".")
    for sentence in sentences[:3]:
        sentence = sentence.strip()
        if len(sentence) < 20:
            continue
        item = {
            "type": "fact",
            "content": sentence,
            "complexity": 0.3,
            **base_meta,
        }
        concepts.append(item)
    if concepts:
        self.processed_hashes.add(content_hash)
    return concepts

