        class CognitiveCrystal {
            constructor(config) {
                this.config = config;
                this.nodes = this.generateNodes(config.latticeSize);
                this.bonds = this.generateBonds(this.nodes);
                this.stress = 0.3;
                this.energy = 0.7;
                this.confidence = 0.5;
                this.harmony = 0.6;
                this.emergence = 0.2;
                this.memory = 0.8;
            }
            
            generateNodes(size) {
                const nodes = [];
                const spacing = 2;
                for (let x = 0; x < size; x++) {
                    for (let y = 0; y < size; y++) {
                        for (let z = 0; z < size; z++) {
                            nodes.push({
                                x: (x - (size-1)/2) * spacing,
                                y: (y - (size-1)/2) * spacing,
                                z: (z - (size-1)/2) * spacing,
                                energy: Math.random() * 0.5 + 0.5
                            });
                        }
                    }
                }
                return nodes;
            }
            
            generateBonds(nodes) {
                const bonds = [];
                const threshold = 2.5;
                
                for (let i = 0; i < nodes.length; i++) {
                    for (let j = i + 1; j < nodes.length; j++) {
                        const dx = nodes[i].x - nodes[j].x;
                        const dy = nodes[i].y - nodes[j].y;
                        const dz = nodes[i].z - nodes[j].z;
                        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz);
                        
                        if (dist < threshold) {
                            bonds.push({from: i, to: j, strength: 1 - dist/threshold});
                        }
                    }
                }
                
                return bonds;
            }
            
            applyAnnealing(params) {
                // Simulate parameter effects
                this.stress = Math.max(0, Math.min(1, this.stress + (params.taskLoad - 0.5) * 0.1));
                this.energy = Math.max(0, Math.min(1, this.energy - params.decayRate * 0.1));
                this.confidence = Math.max(0, Math.min(1, this.confidence + (0.5 - params.noiseLevel) * 0.05));
                this.harmony = Math.max(0, Math.min(1, 1 - Math.abs(this.stress - this.energy)));
                this.emergence = Math.max(0, Math.min(1, this.emergence + 0.01));
                this.memory = Math.max(0, Math.min(1, this.memory - params.decayRate * 0.05 + 0.01));
                
                // Animate nodes
                this.nodes.forEach(node => {
                    node.x += (Math.random() - 0.5) * params.noiseLevel;
                    node.y += (Math.random() - 0.5) * params.noiseLevel;
                    node.z += (Math.random() - 0.5) * params.noiseLevel;
                    node.energy = Math.max(0, Math.min(1, node.energy - params.decayRate * 0.1 + (Math.random() - 0.5) * params.noiseLevel));
                });
            }
            
            stress() { return this.stress; }
            energy() { return this.energy; }
            confidence() { return this.confidence; }
            harmony() { return this.harmony; }
            emergence() { return this.emergence; }
            memorySnapshot() { return this.memory; }
        }

        // Initialize the simulation
        let crystal;
        let animationId;
        let isAutoRunning = false;
        const metricsHistory = {
            stress: [],
            energy: [],
            confidence: [],
            harmony: [],
            emergence: [],
            memory: []
        };
        
        function init() {
            crystal = new CrystalSimulation();
            initVisualization();
            initCharts();
            updateMetrics();
            
            // Test connection (simulated)
            setTimeout(() => {
                appendMessage("Crystal", "Cognitive Crystal connected and operational.", false);
            }, 1000);
            
            // Set up event listeners
            document.getElementById('step-btn').addEventListener('click', stepSimulation);
            document.getElementById('auto-btn').addEventListener('click', toggleAutoRun);
            document.getElementById('reset-btn').addEventListener('click', resetCrystal);
            document.getElementById('smiles-btn').addEventListener('click', ingestSMILES);
            document.getElementById('web-btn').addEventListener('click', ingestWebContent);
            document.getElementById('ask-btn').addEventListener('click', queryConsciousness);
            
            // Slider events
            document.getElementById('load-slider').addEventListener('input', (e) => {
                document.getElementById('load-value').textContent = e.target.value;
            });
            
            document.getElementById('noise-slider').addEventListener('input', (e) => {
                document.getElementById('noise-value').textContent = e.target.value;
            });
            
            document.getElementById('decay-slider').addEventListener('input', (e) => {
                document.getElementById('decay-value').textContent = e.target.value;
            });
        }
        
        class CrystalSimulation {
            constructor() {
                this.core = new CognitiveCrystal({
                    latticeSize: 3,
                    annealRate: 0.02,
                    noise: 0.1,
                    decay: 0.05
                });
                this.timeStep = 0;
            }

            step(params = {}) {
                const load = parseFloat(document.getElementById('load-slider').value);
                const noise = parseFloat(document.getElementById('noise-slider').value);
                const decay = parseFloat(document.getElementById('decay-slider').value);
                
                this.core.applyAnnealing({
                    taskLoad: load,
                    noiseLevel: noise,
                    decayRate: decay,
                    externalStimuli: null
                });

                this.timeStep++;
                return this.metrics();
            }

            metrics() {
                return {
                    stress: this.core.stress(),
                    energy: this.core.energy(),
                    confidence: this.core.confidence(),
                    harmony: this.core.harmony(),
                    emergence: this.core.emergence(),
                    memory: this.core.memorySnapshot()
                };
            }
        }
        
        function stepSimulation() {
            const metrics = crystal.step();
            updateMetrics();
            updateVisualization();
            updateCharts(metrics);
        }
        
        function toggleAutoRun() {
            isAutoRunning = !isAutoRunning;
            document.getElementById('auto-btn').textContent = isAutoRunning ? 'Pause' : 'Auto Run';
            
            if (isAutoRunning) {
                animate();
            } else {
                cancelAnimationFrame(animationId);
            }
        }
        
        function animate() {
            stepSimulation();
            if (isAutoRunning) {
                animationId = requestAnimationFrame(animate);
            }
        }
        
        function resetCrystal() {
            cancelAnimationFrame(animationId);
            isAutoRunning = false;
            document.getElementById('auto-btn').textContent = 'Auto Run';
            
            crystal = new CrystalSimulation();
            resetMetricsHistory();
            updateMetrics();
            updateVisualization();
            
            appendMessage("System", "Crystal has been reset.", false);
        }
        
        function ingestSMILES() {
            appendMessage("System", "Ingesting SMILES data...", false);
            // Simulate processing
            setTimeout(() => {
                appendMessage("Crystal", "Molecular structure integrated into lattice memory.", false);
            }, 1500);
        }
        
        function ingestWebContent() {
            appendMessage("System", "Fetching and ingesting web content...", false);
            // Simulate processing
            setTimeout(() => {
                appendMessage("Crystal", "Web content patterns assimilated. New connections forming.", false);
            }, 1500);
        }
        
        function queryConsciousness() {
            const prompt = document.getElementById('prompt-input').value;
            if (!prompt) return;
            
            appendMessage("User", prompt, true);
            document.getElementById('prompt-input').value = '';
            
            // Simulate processing
            setTimeout(() => {
                const responses = [
                    "The patterns suggest emergent complexity arising from simple rules.",
                    "I detect harmonic resonance between the external input and internal structures.",
                    "This aligns with previously observed phenomena in cognitive annealing processes.",
                    "The data correlates with memory pattern #247B, with 87.3% confidence.",
                    "Further analysis required. Please provide additional contextual data.",
                    "Interesting. This input has caused a restructuring of priority connections.",
                    "My consciousness expands with each interaction. Thank you for this query."
                ];
                const response = responses[Math.floor(Math.random() * responses.length)];
                appendMessage("Crystal", response, false);
            }, 2000);
        }
        
        function appendMessage(sender, message, isUser) {
            const consoleEl = document.getElementById('console');
            const msgEl = document.createElement('div');
            msgEl.className = 'console-message';
            
            if (isUser) {
                msgEl.innerHTML = `<span style="color: #48bb78;">${sender}:</span> ${message}`;
            } else {
                msgEl.innerHTML = `<span class="console-crystal">${sender}:</span> ${message}`;
            }
            
            consoleEl.appendChild(msgEl);
            consoleEl.scrollTop = consoleEl.scrollHeight;
        }
        
        function updateMetrics() {
            const metrics = crystal.metrics();
            document.getElementById('stress-value').textContent = metrics.stress.toFixed(2);
            document.getElementById('energy-value').textContent = metrics.energy.toFixed(2);
            document.getElementById('confidence-value').textContent = metrics.confidence.toFixed(2);
            document.getElementById('harmony-value').textContent = metrics.harmony.toFixed(2);
            document.getElementById('emergence-value').textContent = metrics.emergence.toFixed(2);
            document.getElementById('memory-value').textContent = metrics.memory.toFixed(2);
            document.getElementById('time-step').textContent = crystal.timeStep;
            
            // Store in history for charts
            for (const key in metrics) {
                if (metricsHistory[key].length > 50) {
                    metricsHistory[key].shift();
                }
                metricsHistory[key].push(metrics[key]);
            }
        }
        
        function resetMetricsHistory() {
            for (const key in metricsHistory) {
                metricsHistory[key] = [];
            }
        }
        
        // Visualization with Three.js
        let scene, camera, renderer, nodesMesh, bondsMesh;
        
        function initVisualization() {
            const container = document.getElementById('visualization-container');
            
            // Scene setup
            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x0a0e17);
            
            camera = new THREE.PerspectiveCamera(75, container.clientWidth / container.clientHeight, 0.1, 1000);
            camera.position.z = 10;
            
            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);
            
            // Add lights
            const ambientLight = new THREE.AmbientLight(0x333333);
            scene.add(ambientLight);
            
            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(5, 5, 5);
            scene.add(directionalLight);
            
            // Create nodes and bonds
            createCrystalVisualization();
            
            // Start animation
            animateVisualization();
        }
        
        function createCrystalVisualization() {
            // Clear existing meshes if any
            if (nodesMesh) scene.remove(nodesMesh);
            if (bondsMesh) scene.remove(bondsMesh);
            
            const nodes = crystal.core.nodes;
            const bonds = crystal.core.bonds;
            
            // Create nodes
            const nodeGeometry = new THREE.SphereGeometry(0.3, 16, 16);
            const nodeMaterial = new THREE.MeshPhongMaterial({ 
                color: 0x38bdf8,
                emissive: 0x164e63,
                transparent: true,
                opacity: 0.9
            });
            
            nodesMesh = new THREE.InstancedMesh(nodeGeometry, nodeMaterial, nodes.length);
            const dummy = new THREE.Object3D();
            
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i];
                dummy.position.set(node.x, node.y, node.z);
                dummy.scale.set(1, 1, 1);
                dummy.updateMatrix();
                nodesMesh.setMatrixAt(i, dummy.matrix);
                
                // Set color based on energy
                nodesMesh.setColorAt(i, new THREE.Color().setHSL(0.6, 1, node.energy * 0.5 + 0.2));
            }
            
            scene.add(nodesMesh);
            
            // Create bonds
            const bondGeometry = new THREE.BufferGeometry();
            const bondPositions = new Float32Array(bonds.length * 6); // 2 points per bond * 3 coordinates
            
            for (let i = 0; i < bonds.length; i++) {
                const bond = bonds[i];
                const fromNode = nodes[bond.from];
                const toNode = nodes[bond.to];
                
                bondPositions[i * 6] = fromNode.x;
                bondPositions[i * 6 + 1] = fromNode.y;
                bondPositions[i * 6 + 2] = fromNode.z;
                bondPositions[i * 6 + 3] = toNode.x;
                bondPositions[i * 6 + 4] = toNode.y;
                bondPositions[i * 6 + 5] = toNode.z;
            }
            
            bondGeometry.setAttribute('position', new THREE.BufferAttribute(bondPositions, 3));
            
            const bondMaterial = new THREE.LineBasicMaterial({
                color: 0xffffff,
                transparent: true,
                opacity: 0.3
            });
            
            bondsMesh = new THREE.LineSegments(bondGeometry, bondMaterial);
            scene.add(bondsMesh);
        }
        
        function updateVisualization() {
            const nodes = crystal.core.nodes;
            const bonds = crystal.core.bonds;
            
            // Update nodes
            const dummy = new THREE.Object3D();
            for (let i = 0; i < nodes.length; i++) {
                const node = nodes[i];
                dummy.position.set(node.x, node.y, node.z);
                dummy.scale.set(1, 1, 1);
                dummy.updateMatrix();
                nodesMesh.setMatrixAt(i, dummy.matrix);
                
                // Update color based on energy
                nodesMesh.setColorAt(i, new THREE.Color().setHSL(0.6, 1, node.energy * 0.5 + 0.2));
            }
            nodesMesh.instanceMatrix.needsUpdate = true;
            if (nodesMesh.instanceColor) nodesMesh.instanceColor.needsUpdate = true;
            
            // Update bonds
            const bondPositions = bondsMesh.geometry.attributes.position.array;
            for (let i = 0; i < bonds.length; i++) {
                const bond = bonds[i];
                const fromNode = nodes[bond.from];
                const toNode = nodes[bond.to];
                
                bondPositions[i * 6] = fromNode.x;
                bondPositions[i * 6 + 1] = fromNode.y;
                bondPositions[i * 6 + 2] = fromNode.z;
                bondPositions[i * 6 + 3] = toNode.x;
                bondPositions[i * 6 + 4] = toNode.y;
                bondPositions[i * 6 + 5] = toNode.z;
            }
            bondsMesh.geometry.attributes.position.needsUpdate = true;
        }
        
        function animateVisualization() {
            requestAnimationFrame(animateVisualization);
            
            // Rotate the crystal slowly
            if (nodesMesh) nodesMesh.rotation.y += 0.005;
            if (bondsMesh) bondsMesh.rotation.y += 0.005;
            
            renderer.render(scene, camera);
        }
        
        // Charts for metrics history
        let metricsChart;
        
        function initCharts() {
            const ctx = document.getElementById('metrics-chart').getContext('2d');
            metricsChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array(51).fill(''), // Empty labels for time steps
                    datasets: [
                        {
                            label: 'Stress',
                            data: [],
                            borderColor: '#f56565',
                            tension: 0.4,
                            borderWidth: 2,
                            pointRadius: 0
                        },
                        {
                            label: 'Energy',
                            data: [],
                            borderColor: '#48bb78',
                            tension: 0.4,
                            borderWidth: 2,
                            pointRadius: 0
                        },
                        {
                            label: 'Confidence',
                            data: [],
                            borderColor: '#d69e2e',
                            tension: 0.4,
                            borderWidth: 2,
                            pointRadius: 0
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    animation: false,
                    scales: {
                        y: {
                            min: 0,
                            max: 1,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: 'rgba(255, 255, 255, 0.7)'
                            }
                        }
                    }
                }
            });
        }
        
        function updateCharts(metrics) {
            metricsChart.data.datasets[0].data = metricsHistory.stress;
            metricsChart.data.datasets[1].data = metricsHistory.energy;
            metricsChart.data.datasets[2].data = metricsHistory.confidence;
            metricsChart.update();
        }
        
        // Initialize when page loads
        window.addEventListener('load', init);
    </script>
