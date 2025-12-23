// ... Previous
private verbMotions: any = VERB_ACTIONS;  // Copy from Py or hardcode

private applyVerbMotion(targets: NodeTarget[], verbs: string[]): void {
    verbs.forEach(verb => {
        const action = this.verbMotions[verb];
        if (!action) return;
        switch (action.type) {
            case 'translate':
                targets.forEach(t => t.position.add(new THREE.Vector3(...action.direction).multiplyScalar(action.speed)));
                break;
            case 'pulse':
                targets.forEach(t => t.intensity = (t.intensity || 1) * action.intensity + Math.sin(Date.now() * action.freq) * 0.5);
                break;
            case 'rotate':
                const matrix = new THREE.Matrix4().makeRotationAxis(new THREE.Vector3(...action.axis), action.speed * 0.01);
                targets.forEach(t => t.position.applyMatrix4(matrix));
                break;
            case 'attract':
                // Assume target is last shape; simple converge to center
                const center = new THREE.Vector3();  // Compute from next noun targets
                targets.forEach(t => t.position.lerp(center, 0.1));
                break;
            case 'scale':
                targets.forEach(t => t.position.multiplyScalar(action.factor));
                break;
        }
    });
}

// In updateFrame, after lerp:
this.applyVerbMotion(updatedTargets, currentVerbs);  # Assume verbs from context

// In process... add verbs to targets
