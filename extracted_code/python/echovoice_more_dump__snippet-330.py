function initCrystalVisualization(container) {
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Bind nodes & bonds from crystal core
    const nodeGeom = new THREE.SphereGeometry(0.2, 12, 12);
    const nodeMat = new THREE.MeshPhongMaterial({ color: 0x38bdf8, transparent: true });
    const nodesMesh = new THREE.InstancedMesh(nodeGeom, nodeMat, crystal.core.nodes.length);

    scene.add(nodesMesh);
    // … update loop attaches to crystal.core.nodes positions

    return { scene, camera, renderer, nodesMesh };
