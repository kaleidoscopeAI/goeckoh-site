const node = svc.nodeStates[0];
const mods1 = (await Promise.resolve()).then(() => require("../src/services/emotionIntegrators").computeModulators(node));
