const phrases = [
"I need a break.",
"I am okay.",
"I feel overwhelmed.",
"I am excited.",
"I want to stim.",
"I am safe right now.",
"I can breathe slow.",
"I am listening."
];
Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.
62/95let currentPhrase = "";
function renderPhrases() {
const container = document.getElementById("phrase-buttons");
container.innerHTML = "";
phrases.forEach((p) => {
const btn = document.createElement("button");
btn.className = "btn btn-small";
btn.textContent = p;
btn.onclick = () => pickPhrase(p);
container.appendChild(btn);
});
}
function pickPhrase(p) {
currentPhrase = p;
document.getElementById("prompt-text").textContent = "Practicing: \"" + p + "\"";
log("Phrase picked: " + p);
notifyApp("phrase:" + p);
}
function nextPrompt() {
const idx = Math.floor(Math.random() * phrases.length);
pickPhrase(phrases[idx]);
}
function log(msg) {
const logEl = document.getElementById("log");
const line = document.createElement("div");
line.className = "log-line";
const ts = new Date().toLocaleTimeString();
line.innerHTML = "<span>[" + ts + "]</span> " + msg;
logEl.prepend(line);
}
function notifyApp(message) {
if (window.webkit && window.webkit.messageHandlers && window.webkit.messageHandlers.echoGame) {
window.webkit.messageHandlers.echoGame.postMessage(message);
}
}
function updateMetrics(arousal, valence, temp, coh) {
document.getElementById("m-arousal").textContent = arousal.toFixed(2);
document.getElementById("m-valence").textContent = valence.toFixed(2);
document.getElementById("m-temp").textContent = temp.toFixed(3);
document.getElementById("m-coh").textContent = coh.toFixed(3);
}
function setStateLabel(text) {
document.getElementById("state-label").textContent = text;
}
renderPhrases();
nextPrompt();
setStateLabel("Ready");
