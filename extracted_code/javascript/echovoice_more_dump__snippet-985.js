+ const files = fs.readdirSync(W_DIR).filter((f) => f.endsWith(".json"));
+ const items = files.map((f) => JSON.parse(fs.readFileSync(path.join(W_DIR, f), "utf8")));
