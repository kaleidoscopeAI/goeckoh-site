private matchToImagePoints(targets: NodeTarget[], points: number[][]): void {
    points.forEach((p, i) => {
        targets[i % targets.length].position.set(...p);
    });
