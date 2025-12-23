const unsub = service.subscribe((cs) => setConstructs(cs));
// initial update from provider
const nodes = nodesProvider ? nodesProvider() : [];
