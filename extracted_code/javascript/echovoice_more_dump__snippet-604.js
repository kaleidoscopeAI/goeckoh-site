export function simulationStep() {
const { e } = useEmotionalActuation();
const speciesInputs = mapEToSpecies(e, Simulation.nodes.length);
for (let i = 0; i < Simulation.nodes.length; i++) Simulation.nodes[i].externalInput = speciesInputs[i];
