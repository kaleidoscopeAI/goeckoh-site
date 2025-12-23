function updateStats(stats) {
const labels = [];
const values = [];
const rows = [];
for (const [phrase, s] of Object.entries(stats)) {
