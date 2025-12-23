// Monitor DSP
UI.monitorOn = document.getElementById("monitorOn");
UI.monitorGain = document.getElementById("monitorGain");
UI.gainVal = document.getElementById("gainVal");

let monitorGainNode = null;
const applyMonitor = () => {
  if (!monitorGainNode) return;
  const g = Number(UI.monitorGain.value);
  UI.gainVal.textContent = g.toFixed(2);
  monitorGainNode.gain.value = UI.monitorOn.checked ? g : 0.0;
};
UI.monitorOn.oninput = UI.monitorGain.oninput = applyMonitor;
