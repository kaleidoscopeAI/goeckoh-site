export function updateHomeostasisAndMaturity(node: NodeState, dtSeconds: number) {
const s = node.species;
const h = node.homeostasis;
const rho = DEFAULT_PARAMS.homeoRho;
for (let i = 0; i < s.length; i++) {
