    const metrics = crystal.step();

    // Update the visualization
    updateVisualization(viz, crystal);

    // Update charts and metrics display (if any)
    updateCharts(metrics);

    renderer.render(scene, camera);

