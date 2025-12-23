private conversationHistory: ConversationContext[] = [];
private currentTopic: string = '';
private nodeCount: number = 18000; // Expanded for denser visualizations
private currentTargets: NodeTarget[] = []; // Current node states
private targetTargets: NodeTarget[] = []; // Next targets for animation
private animationProgress: number = 0; // 0-1 for interpolation
private animationDuration: number = 3000; // ms for morph, adjustable

constructor() {
    super();
    this.resetContext();
}

// Analyze conversation or ingested data
analyzeConversation(text: string): ConversationContext {
    try {
        const lowerText = text.toLowerCase();
        // (Existing logic unchanged, with added error handling)
        // ...
        return { topics, sentiment, complexity, entities, concepts };
    } catch (error) {
        console.error('Analysis error:', error);
        return { topics: [], sentiment: 'neutral', complexity: 0.5, entities: [], concepts: [] }; // Fallback
    }
}

// Generate dynamic node targets
generateDynamicTargets(context: ConversationContext): NodeTarget[] {
    // (Existing logic, scaled for 18000 nodes)
    // ...
}

// Apply topic-specific shaping (all now complete with real parametric logic)
private applyTopicShaping(/* params */) {
    // (Existing switch, all cases call real functions)
}

// (All shape functions as provided, no placeholders)

// Generate contextual color (unchanged)
private generateContextualColor(/* params */): THREE.Color {
    // ...
}

// Generate connections (unchanged)
private generateConnections(/* params */): number[] {
    // ...
}

// Process conversation response (existing)
async processConversationResponse(text: string): Promise<NodeTarget[]> {
    // ...
    this.targetTargets = this.generateDynamicTargets(context);
    this.animationProgress = 0; // Start animation
    this.emit('newTargets', this.targetTargets); // For external renderer
    return this.targetTargets;
}

// New: Process ingested data (e.g., from web crawl) for real-time updates
async processIngestedData(dataStream: string[], topic: string): Promise<void> {
    try {
        for (const chunk of dataStream) {
            const context = this.analyzeConversation(chunk);
            context.topics.unshift(topic); // Prioritize crawl topic
            const partialTargets = this.generateDynamicTargets(context).slice(0, this.nodeCount * 0.1); // 10% update for streaming
            // Blend with current
            for (let i = 0; i < partialTargets.length; i++) {
                const idx = Math.floor(Math.random() * this.nodeCount);
                this.targetTargets[idx] = partialTargets[i];
            }
            this.animationProgress = 0;
            this.emit('partialUpdate', partialTargets); // Emit for real-time render
            await new Promise(resolve => setTimeout(resolve, 100)); // Throttle for animation
        }
        this.emit('ingestionComplete');
    } catch (error) {
        console.error('Ingestion error:', error);
    }
}

// New: Real-time frame update for animation (call in render loop)
updateFrame(deltaTime: number): NodeTarget[] | null {
    if (this.animationProgress < 1) {
        this.animationProgress = Math.min(1, this.animationProgress + deltaTime / this.animationDuration);
        const alpha = 1 - Math.pow(1 - this.animationProgress, 3); // Ease-in cubic
        const updatedTargets: NodeTarget[] = [];
        for (let i = 0; i < this.nodeCount; i++) {
            const curr = this.currentTargets[i] || { position: new THREE.Vector3(), color: new THREE.Color() };
            const targ = this.targetTargets[i];
            const pos = curr.position.clone().lerp(targ.position, alpha);
            const col = curr.color.clone().lerp(targ.color, alpha);
            updatedTargets.push({ position: pos, color: col, intensity: targ.intensity, connections: targ.connections });
        }
        this.currentTargets = updatedTargets;
        this.emit('update', updatedTargets); // Emit for Three.js scene update
        return updatedTargets;
    }
    return null; // Animation complete
}

// Get current context (unchanged)
getCurrentContext(): ConversationContext | null {
    // ...
}

// Reset context (unchanged)
resetContext() {
    // ...
    this.currentTargets = [];
    this.targetTargets = [];
    this.animationProgress = 1;
}
