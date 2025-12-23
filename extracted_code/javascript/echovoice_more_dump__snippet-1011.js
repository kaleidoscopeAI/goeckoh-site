class DynamicVisualizationService {
    private conversationHistory: ConversationContext[] = [];
    private currentTopic: string = '';
    private nodeCount: number = 8000;

    // Analyze conversation and extract semantic meaning
    analyzeConversation(text: string): ConversationContext {
        const lowerText = text.toLowerCase();

        // Extract topics and entities
        const topics: string[] = [];
        const entities: string[] = [];
        const concepts: string[] = [];

        // Simple keyword extraction (in production, use NLP)
        for (const [topic, shape] of Object.entries(TOPIC_SHAPES)) {
            if (lowerText.includes(topic)) {
                topics.push(topic);
                entities.push(topic);
            }
        }

        // Extract abstract concepts
        const conceptKeywords = ['think', 'feel', 'know', 'understand', 'create', 'learn', 'explore', 'discover'];
        for (const concept of conceptKeywords) {
            if (lowerText.includes(concept)) {
                concepts.push(concept);
            }
        }

        // Determine sentiment
        const positiveWords = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love', 'like', 'happy', 'joy'];
        const negativeWords = ['bad', 'terrible', 'awful', 'hate', 'sad', 'angry', 'frustrated'];

        let positiveScore = 0;
        let negativeScore = 0;

        for (const word of positiveWords) {
            if (lowerText.includes(word)) positiveScore++;
        }
        for (const word of negativeWords) {
            if (lowerText.includes(word)) negativeScore++;
        }

        let sentiment: 'positive' | 'negative' | 'neutral' = 'neutral';
        if (positiveScore > negativeScore) sentiment = 'positive';
        else if (negativeScore > positiveScore) sentiment = 'negative';

        // Calculate complexity based on word variety and length
        const words = text.split(' ');
        const uniqueWords = new Set(words.map(w => w.toLowerCase()));
        const complexity = Math.min(1.0, (uniqueWords.size / words.length) * (text.length / 100));

        return {
            topics,
            sentiment,
            complexity,
            entities,
            concepts
        };
    }

    // Generate dynamic node targets based on conversation context
    generateDynamicTargets(context: ConversationContext): NodeTarget[] {
        const targets: NodeTarget[] = [];
        const primaryTopic = context.topics[0] || 'abstract';

        // Base configuration
        const numNodes = this.nodeCount;
        const spread = 400;
        const height = 300;

        for (let i = 0; i < numNodes; i++) {
            const t = (i / numNodes) * Math.PI * 4; // Multiple spirals
            let radius = Math.sqrt(i / numNodes) * spread;

            // Modify radius and height based on sentiment and complexity
            if (context.sentiment === 'positive') {
                radius *= 1.2;
                height *= 1.3;
            } else if (context.sentiment === 'negative') {
                radius *= 0.8;
                height *= 0.7;
            }

            // Apply topic-specific shaping
            const shapedPosition = this.applyTopicShaping(
                new THREE.Vector3(
                    Math.cos(t) * radius,
                    Math.sin(t) * radius,
                    (Math.random() - 0.5) * height
                ),
                primaryTopic,
                context,
                i,
                numNodes
            );

            // Generate colors based on context
            const color = this.generateContextualColor(context, i, numNodes);

            // Add some randomness for organic feel
            shapedPosition.x += (Math.random() - 0.5) * 50;
            shapedPosition.y += (Math.random() - 0.5) * 50;
            shapedPosition.z += (Math.random() - 0.5) * 100;

            targets.push({
                position: shapedPosition,
                color,
                intensity: context.complexity,
                connections: this.generateConnections(i, numNodes, context)
            });
        }

        return targets;
    }

    // Apply topic-specific shaping to node positions
    private applyTopicShaping(
        basePosition: THREE.Vector3,
        topic: string,
        context: ConversationContext,
        index: number,
        totalNodes: number
    ): THREE.Vector3 {
        const position = basePosition.clone();

        switch (topic) {
            case 'elephant':
                return this.shapeElephant(position, index, totalNodes);
            case 'tree':
                return this.shapeTree(position, index, totalNodes);
            case 'brain':
                return this.shapeBrain(position, index, totalNodes);
            case 'network':
                return this.shapeNetwork(position, index, totalNodes);
            case 'consciousness':
                return this.shapeConsciousness(position, index, totalNodes);
            case 'cat':
                return this.shapeCat(position, index, totalNodes);
            case 'flower':
                return this.shapeFlower(position, index, totalNodes);
            case 'mountain':
                return this.shapeMountain(position, index, totalNodes);
            default:
                return this.shapeAbstract(position, index, totalNodes, context);
        }
    }

    // Shape nodes to form an elephant
    private shapeElephant(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
        const progress = index / totalNodes;

        if (progress < 0.3) {
            // Trunk (curved shape)
            const trunkProgress = progress / 0.3;
            position.x *= 0.3;
            position.y = position.y * 0.5 + Math.sin(trunkProgress * Math.PI) * 100;
            position.z *= 0.8;
        } else if (progress < 0.6) {
            // Body (oval shape)
            const bodyProgress = (progress - 0.3) / 0.3;
            const angle = bodyProgress * Math.PI * 2;
            position.x = Math.cos(angle) * 150;
            position.y = Math.sin(angle) * 100;
            position.z *= 0.6;
        } else {
            // Ears and legs
            const earProgress = (progress - 0.6) / 0.4;
            const side = earProgress < 0.5 ? -1 : 1;
            position.x = side * (150 + Math.sin(earProgress * Math.PI) * 80);
            position.y *= 0.8;
            position.z = Math.cos(earProgress * Math.PI) * 50;
        }

        return position;
    }

    // Shape nodes to form a tree
    private shapeTree(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
        const progress = index / totalNodes;

        if (progress < 0.7) {
            // Trunk (vertical)
            position.x *= 0.3;
            position.y *= 0.3;
            position.z = progress * 200 - 100;
        } else {
            // Branches and leaves (spiral outward)
            const branchProgress = (progress - 0.7) / 0.3;
            const branchRadius = branchProgress * 120;
            const branchHeight = (1 - branchProgress) * 150;
            const angle = branchProgress * Math.PI * 6;

            position.x = Math.cos(angle) * branchRadius;
            position.y = Math.sin(angle) * branchRadius;
            position.z = branchHeight;
        }

        return position;
    }

    // Shape nodes to form a brain
    private shapeBrain(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
        const progress = index / totalNodes;

        // Create brain-like folded structure
        const folds = 8;
        const foldRadius = 100;
        const foldHeight = Math.sin(progress * Math.PI * folds) * 30;

        position.x *= 1.2;
        position.y *= 1.2;
        position.z = foldHeight + (Math.random() - 0.5) * 60;

        return position;
    }

    // Shape nodes to form a network/web
    private shapeNetwork(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
        const progress = index / totalNodes;

        // Create web-like structure with multiple connection points
        const webCenters = 5;
        const centerIndex = Math.floor(progress * webCenters);
        const centerAngle = (centerIndex / webCenters) * Math.PI * 2;
        const distance = Math.sin(progress * Math.PI * 8) * 80;

        position.x = Math.cos(centerAngle) * distance + Math.cos(progress * Math.PI * 16) * 40;
        position.y = Math.sin(centerAngle) * distance + Math.sin(progress * Math.PI * 16) * 40;
        position.z *= 0.8;

        return position;
    }

    // Shape nodes to represent consciousness (spiral galaxy-like)
    private shapeConsciousness(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
        const progress = index / totalNodes;

        // Create spiral galaxy pattern
        const spiralArms = 3;
        const armIndex = Math.floor(progress * spiralArms);
        const armProgress = (progress * spiralArms) % 1;
        const radius = armProgress * 200;
        const angle = armProgress * Math.PI * 4 + (armIndex * Math.PI * 2 / spiralArms);

        position.x = Math.cos(angle) * radius;
        position.y = Math.sin(angle) * radius;
        position.z = Math.sin(progress * Math.PI * 4) * 100;

        return position;
    }

    // Abstract shaping for unrecognized topics
    private shapeAbstract(position: THREE.Vector3, index: number, totalNodes: number, context: ConversationContext): THREE.Vector3 {
        const progress = index / totalNodes;

        // Create organic, flowing patterns based on complexity
        const complexity = context.complexity;
        const flow = Math.sin(progress * Math.PI * 8 * complexity) * 50;

        position.x += flow;
        position.y += Math.cos(progress * Math.PI * 6 * complexity) * 40;
        position.z += Math.sin(progress * Math.PI * 4 * complexity) * 60;

        return position;
    }

    // Additional shape functions for other topics
    private shapeCat(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
        const progress = index / totalNodes;

        if (progress < 0.4) {
            // Body (elongated oval)
            position.x *= 1.5;
            position.y *= 0.8;
            position.z *= 0.6;
        } else if (progress < 0.7) {
            // Head
            const headProgress = (progress - 0.4) / 0.3;
            position.x = (headProgress - 0.5) * 80;
            position.y *= 0.6;
            position.z *= 0.8;
        } else {
            // Tail (curved)
            const tailProgress = (progress - 0.7) / 0.3;
            position.x = Math.sin(tailProgress * Math.PI) * 60;
            position.y = Math.cos(tailProgress * Math.PI) * 40;
            position.z *= 0.5;
        }

        return position;
    }

    private shapeFlower(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
        const progress = index / totalNodes;

        // Create flower-like pattern with petals
        const petals = 6;
        const petalIndex = Math.floor(progress * petals);
        const petalProgress = (progress * petals) % 1;
        const petalRadius = 80;

        const angle = petalIndex * (Math.PI * 2 / petals) + petalProgress * (Math.PI / 3);
        position.x = Math.cos(angle) * petalRadius * (0.5 + Math.sin(petalProgress * Math.PI) * 0.5);
        position.y = Math.sin(angle) * petalRadius * (0.5 + Math.sin(petalProgress * Math.PI) * 0.5);
        position.z *= 0.3;

        return position;
    }

    private shapeMountain(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
        const progress = index / totalNodes;

        // Create mountain range silhouette
        const peaks = 4;
        const peakIndex = Math.floor(progress * peaks);
        const peakProgress = (progress * peaks) % 1;

        if (peakProgress < 0.5) {
            // Ascending slope
            position.x = (peakIndex * 100) + (peakProgress * 50);
            position.y *= 0.8;
            position.z = Math.sin(peakProgress * Math.PI) * 120;
        } else {
            // Descending slope
            const descendProgress = (peakProgress - 0.5) * 2;
            position.x = (peakIndex * 100) + 50 + (descendProgress * 50);
            position.y *= 0.8;
            position.z = Math.sin((1 - descendProgress) * Math.PI) * 120;
        }

        return position;
    }

    // Generate contextual colors based on conversation content
    private generateContextualColor(context: ConversationContext, index: number, totalNodes: number): THREE.Color {
        const progress = index / totalNodes;

        // Base color influenced by sentiment
        let hue = 0.6; // Blue default

        if (context.sentiment === 'positive') {
            hue = 0.3; // Green-yellow
        } else if (context.sentiment === 'negative') {
            hue = 0.0; // Red
        }

        // Add variation based on complexity and topics
        hue += context.complexity * 0.2;
        hue += Math.sin(progress * Math.PI * 4) * 0.1;

        // Topic-specific color adjustments
        if (context.topics.includes('elephant')) {
            hue = 0.1; // Gray-brown
        } else if (context.topics.includes('tree')) {
            hue = 0.3; // Green
        } else if (context.topics.includes('brain')) {
            hue = 0.7; // Purple
        } else if (context.topics.includes('consciousness')) {
            hue = 0.8; // Light blue-purple
        }

        const saturation = 0.6 + context.complexity * 0.4;
        const lightness = 0.4 + Math.sin(progress * Math.PI * 2) * 0.3;

        return new THREE.Color().setHSL(hue, saturation, lightness);
    }

    // Generate connection patterns for nodes
    private generateConnections(nodeIndex: number, totalNodes: number, context: ConversationContext): number[] {
        const connections: number[] = [];
        const connectionCount = Math.floor(3 + context.complexity * 5); // 3-8 connections

        for (let i = 0; i < connectionCount; i++) {
            // Create connections to nearby nodes and some distant ones for complexity
            const baseOffset = i * (totalNodes / connectionCount);
            let targetNode = Math.floor((nodeIndex + baseOffset) % totalNodes);

            // Add some randomness for organic feel
            targetNode = (targetNode + Math.floor(Math.random() * 10) - 5) % totalNodes;

            if (targetNode !== nodeIndex && !connections.includes(targetNode)) {
                connections.push(targetNode);
            }
        }

        return connections;
    }

    // Main function to process conversation and generate visualization
    async processConversationResponse(text: string): Promise<NodeTarget[]> {
        // Analyze the conversation
        const context = this.analyzeConversation(text);

        // Store in history
        this.conversationHistory.push(context);
        if (this.conversationHistory.length > 10) {
            this.conversationHistory.shift(); // Keep only recent context
        }

        // Generate dynamic visualization
        return this.generateDynamicTargets(context);
    }

    // Get current conversation context
    getCurrentContext(): ConversationContext | null {
        return this.conversationHistory.length > 0
            ? this.conversationHistory[this.conversationHistory.length - 1]
            : null;
    }

    // Reset conversation context
    resetContext() {
        this.conversationHistory = [];
        this.currentTopic = '';
    }
