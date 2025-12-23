export function updateHomeostasisAndMaturity(node: NodeState, dtSeconds: number) {
const s = node.species;
const h = node.homeostasis;
const rho = DEFAULT_PARAMS.homeoRho;
