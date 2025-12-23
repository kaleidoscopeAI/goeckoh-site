const files = fs.readdirSync(W_DIR).filter(f => f.endsWith(".json"));
const items = files.map(f => {
const raw = fs.readFileSync(path.join(W_DIR, f), "utf8");
const p = JSON.parse(raw);
