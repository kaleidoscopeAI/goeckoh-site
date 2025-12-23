function setStatus(s) { UI.status.textContent = s; }
function fmt(x, d=3) { return Number.isFinite(x) ? x.toFixed(d) : "—"; }

function updateUI() {
