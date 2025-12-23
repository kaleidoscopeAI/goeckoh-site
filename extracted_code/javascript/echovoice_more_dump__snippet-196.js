async function init() {
    crystal = new CrystalSimulation();

    // Test Ollama connection
    const res = await queryOllama({
        model: OLLAMA_MODEL,
        messages: [{ role: "system", content: "test" }]
    });
    appendMessage("Crystal", res.includes("Error") ? res : "Cognitive Crystal connected.", false);

    const viz = initCrystalVisualization(document.getElementById("visualization-container"));
    // Hook charts, metrics, control panel to crystal.step()
