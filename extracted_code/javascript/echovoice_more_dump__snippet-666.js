const files = fs.readdirSync(EXP_DIR).filter(f => f.endsWith(".json"));
const items = files.map(f => {
const p = JSON.parse(fs.readFileSync(path.join(EXP_DIR, f), "utf8"));
