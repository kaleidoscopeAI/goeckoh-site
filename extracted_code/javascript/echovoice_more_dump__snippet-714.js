const nodes: any[] = req.body.nodes ?? [];
const W: number[][] | undefined = req.body.W;
const constructs: string[] = req.body.constructs ?? [];
const topK = parseInt(String(req.body.topK ?? "10"), 10);
