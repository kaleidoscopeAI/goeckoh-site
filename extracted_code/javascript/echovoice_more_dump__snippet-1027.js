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
        case 'dog':
            return this.shapeDog(position, index, totalNodes);
        case 'bird':
            return this.shapeBird(position, index, totalNodes);
        case 'fish':
            return this.shapeFish(position, index, totalNodes);
        case 'horse':
            return this.shapeHorse(position, index, totalNodes);
        case 'lion':
            return this.shapeLion(position, index, totalNodes);
        case 'tiger':
            return this.shapeTiger(position, index, totalNodes);
        case 'ocean':
            return this.shapeOcean(position, index, totalNodes);
        case 'sun':
            return this.shapeSun(position, index, totalNodes);
        case 'moon':
            return this.shapeMoon(position, index, totalNodes);
        case 'star':
            return this.shapeStar(position, index, totalNodes);
        case 'cloud':
            return this.shapeCloud(position, index, totalNodes);
        case 'computer':
            return this.shapeComputer(position, index, totalNodes);
        case 'phone':
            return this.shapePhone(position, index, totalNodes);
        case 'robot':
            return this.shapeRobot(position, index, totalNodes);
        case 'neural':
            return this.shapeNeural(position, index, totalNodes);
        case 'ai':
            return this.shapeAI(position, index, totalNodes);
        case 'code':
            return this.shapeCode(position, index, totalNodes);
        case 'happy':
            return this.shapeHappy(position, index, totalNodes);
        case 'sad':
            return this.shapeSad(position, index, totalNodes);
        case 'angry':
            return this.shapeAngry(position, index, totalNodes);
        case 'love':
            return this.shapeLove(position, index, totalNodes);
        case 'fear':
            return this.shapeFear(position, index, totalNodes);
        case 'joy':
            return this.shapeJoy(position, index, totalNodes);
        case 'peace':
            return this.shapePeace(position, index, totalNodes);
        case 'intelligence':
            return this.shapeIntelligence(position, index, totalNodes);
        case 'creativity':
            return this.shapeCreativity(position, index, totalNodes);
        case 'knowledge':
            return this.shapeKnowledge(position, index, totalNodes);
        case 'wisdom':
            return this.shapeWisdom(position, index, totalNodes);
        case 'energy':
            return this.shapeEnergy(position, index, totalNodes);
        case 'time':
            return this.shapeTime(position, index, totalNodes);
        case 'space':
            return this.shapeSpace(position, index, totalNodes);
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

// Shape nodes to form a cat
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

// Shape nodes to form a flower
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

// Shape nodes to form a mountain
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

// Shape nodes to form a dog
private shapeDog(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    if (progress < 0.4) {
        // Body (elongated with legs)
        position.x *= 1.2;
        position.y *= 0.7;
        position.z *= 0.5;
    } else if (progress < 0.7) {
        // Head with ears
        const headProgress = (progress - 0.4) / 0.3;
        position.x = (headProgress - 0.5) * 70;
        position.y *= 0.5;
        position.z = Math.sin(headProgress * Math.PI) * 40;
    } else {
        // Tail (wagging curve)
        const tailProgress = (progress - 0.7) / 0.3;
        position.x = Math.cos(tailProgress * Math.PI * 2) * 50;
        position.y = Math.sin(tailProgress * Math.PI) * 30;
        position.z *= 0.4;
    }

    return position;
}

// Shape nodes to form a bird
private shapeBird(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    if (progress < 0.5) {
        // Body and wings (spread out)
        const wingProgress = progress / 0.5;
        const side = wingProgress < 0.5 ? -1 : 1;
        position.x = side * (100 + Math.cos(wingProgress * Math.PI) * 60);
        position.y *= 0.6;
        position.z *= 0.7;
    } else {
        // Beak and tail
        const tailProgress = (progress - 0.5) / 0.5;
        position.x = Math.sin(tailProgress * Math.PI) * 40;
        position.y = Math.cos(tailProgress * Math.PI) * 50;
        position.z *= 0.5;
    }

    return position;
}

// Shape nodes to form a fish
private shapeFish(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    if (progress < 0.6) {
        // Body (streamlined oval)
        position.x *= 1.8;
        position.y *= 0.4;
        position.z *= 0.3;
    } else {
        // Fins and tail (wavy)
        const finProgress = (progress - 0.6) / 0.4;
        position.x = Math.cos(finProgress * Math.PI * 3) * 50;
        position.y = Math.sin(finProgress * Math.PI * 3) * 30;
        position.z *= 0.6;
    }

    return position;
}

// Shape nodes to form a horse
private shapeHorse(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    if (progress < 0.5) {
        // Body and legs (elongated)
        position.x *= 1.4;
        position.y *= 0.9;
        position.z *= 0.7;
    } else if (progress < 0.8) {
        // Head and mane
        const headProgress = (progress - 0.5) / 0.3;
        position.x = (headProgress - 0.5) * 90;
        position.y *= 0.7;
        position.z = Math.sin(headProgress * Math.PI) * 50;
    } else {
        // Tail
        const tailProgress = (progress - 0.8) / 0.2;
        position.x = Math.sin(tailProgress * Math.PI) * 70;
        position.y = Math.cos(tailProgress * Math.PI) * 40;
        position.z *= 0.5;
    }

    return position;
}

// Shape nodes to form a lion
private shapeLion(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    if (progress < 0.4) {
        // Body
        position.x *= 1.3;
        position.y *= 0.8;
        position.z *= 0.6;
    } else if (progress < 0.7) {
        // Mane (circular fluff)
        const maneProgress = (progress - 0.4) / 0.3;
        const angle = maneProgress * Math.PI * 2;
        position.x = Math.cos(angle) * 60;
        position.y = Math.sin(angle) * 60;
        position.z *= 0.7;
    } else {
        // Tail
        const tailProgress = (progress - 0.7) / 0.3;
        position.x = Math.sin(tailProgress * Math.PI) * 50;
        position.y = Math.cos(tailProgress * Math.PI) * 30;
        position.z *= 0.5;
    }

    return position;
}

// Shape nodes to form a tiger
private shapeTiger(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    if (progress < 0.4) {
        // Body (muscular elongated)
        position.x *= 1.6;
        position.y *= 0.7;
        position.z *= 0.5;
    } else if (progress < 0.7) {
        // Head with stripes simulation
        const headProgress = (progress - 0.4) / 0.3;
        position.x = (headProgress - 0.5) * 80;
        position.y *= 0.6;
        position.z = Math.cos(headProgress * Math.PI * 4) * 20; // Stripe-like waves
    } else {
        // Tail
        const tailProgress = (progress - 0.7) / 0.3;
        position.x = Math.sin(tailProgress * Math.PI * 2) * 60;
        position.y = Math.cos(tailProgress * Math.PI) * 40;
        position.z *= 0.5;
    }

    return position;
}

// Shape nodes to form an ocean (wave-like)
private shapeOcean(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Wave patterns
    const waveFrequency = 5;
    position.x *= 1.5;
    position.y = Math.sin(progress * Math.PI * waveFrequency) * 50;
    position.z *= 0.2;

    return position;
}

// Shape nodes to form a sun (spherical with rays)
private shapeSun(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Central sphere with radiating rays
    const rays = 8;
    const rayIndex = Math.floor(progress * rays);
    const rayProgress = (progress * rays) % 1;
    const radius = 100;

    const angle = rayIndex * (Math.PI * 2 / rays);
    position.x = Math.cos(angle) * radius * rayProgress;
    position.y = Math.sin(angle) * radius * rayProgress;
    position.z *= 0.4;

    return position;
}

// Shape nodes to form a moon (crescent-like)
private shapeMoon(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Crescent shape
    position.x = Math.cos(progress * Math.PI) * 80;
    position.y = Math.sin(progress * Math.PI) * 80 * (1 - progress);
    position.z *= 0.3;

    return position;
}

// Shape nodes to form a star (pointed)
private shapeStar(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Star points
    const points = 5;
    const pointIndex = Math.floor(progress * points);
    const pointProgress = (progress * points) % 1;
    const radiusInner = 50;
    const radiusOuter = 100;

    const radius = pointIndex % 2 === 0 ? radiusOuter : radiusInner;
    const angle = pointIndex * (Math.PI * 2 / points) + pointProgress * (Math.PI * 2 / points);
    position.x = Math.cos(angle) * radius;
    position.y = Math.sin(angle) * radius;
    position.z *= 0.2;

    return position;
}

// Shape nodes to form a cloud (fluffy clusters)
private shapeCloud(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Clustered fluffy shapes
    const clusters = 6;
    const clusterIndex = Math.floor(progress * clusters);
    const clusterProgress = (progress * clusters) % 1;
    const clusterRadius = 60 + Math.sin(clusterProgress * Math.PI) * 40;

    const angle = clusterIndex * (Math.PI * 2 / clusters);
    position.x = Math.cos(angle) * clusterRadius;
    position.y = Math.sin(angle) * clusterRadius;
    position.z *= 0.4;

    return position;
}

// Shape nodes to form a computer (box-like with screen)
private shapeComputer(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    if (progress < 0.6) {
        // Box body
        position.x *= 0.8;
        position.y *= 1.0;
        position.z *= 0.5;
    } else {
        // Screen and keyboard
        const screenProgress = (progress - 0.6) / 0.4;
        position.x = Math.cos(screenProgress * Math.PI) * 50;
        position.y = Math.sin(screenProgress * Math.PI) * 70;
        position.z *= 0.3;
    }

    return position;
}

// Shape nodes to form a phone (rectangular with screen)
private shapePhone(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Flat rectangular shape
    position.x *= 0.6;
    position.y *= 1.2;
    position.z *= 0.1 + Math.sin(progress * Math.PI * 4) * 0.05; // Slight wave for buttons

    return position;
}

// Shape nodes to form a robot (humanoid)
private shapeRobot(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    if (progress < 0.4) {
        // Torso
        position.x *= 0.8;
        position.y *= 1.0;
        position.z *= 0.6;
    } else if (progress < 0.7) {
        // Arms and legs
        const limbProgress = (progress - 0.4) / 0.3;
        const side = limbProgress < 0.5 ? -1 : 1;
        position.x = side * 60;
        position.y *= 0.8;
        position.z = Math.sin(limbProgress * Math.PI) * 50;
    } else {
        // Head
        position.x *= 0.5;
        position.y *= 0.5;
        position.z = 100;
    }

    return position;
}

// Shape nodes to form a neural (neuron-like)
private shapeNeural(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Dendrites and axon
    const branches = 4;
    const branchIndex = Math.floor(progress * branches);
    const branchProgress = (progress * branches) % 1;
    const branchRadius = branchProgress * 80;
    const angle = branchIndex * (Math.PI * 2 / branches) + branchProgress * Math.PI;

    position.x = Math.cos(angle) * branchRadius;
    position.y = Math.sin(angle) * branchRadius;
    position.z *= 0.5;

    return position;
}

// Shape nodes to form an AI (circuit-like)
private shapeAI(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Circuit board pattern
    position.x = Math.cos(progress * Math.PI * 10) * 100;
    position.y = Math.sin(progress * Math.PI * 10) * 100;
    position.z *= 0.2;

    return position;
}

// Shape nodes to form code (line-like structures)
private shapeCode(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Linear code lines with branches
    position.x = progress * 200 - 100;
    position.y = Math.sin(progress * Math.PI * 5) * 50;
    position.z *= 0.3;

    return position;
}

// Shape nodes to form happy (upward curves)
private shapeHappy(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Smiley-like upward arcs
    position.x = Math.cos(progress * Math.PI) * 100;
    position.y = Math.abs(Math.sin(progress * Math.PI)) * 80;
    position.z *= 0.4;

    return position;
}

// Shape nodes to form sad (downward droop)
private shapeSad(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Frown-like downward arcs
    position.x = Math.cos(progress * Math.PI) * 100;
    position.y = -Math.abs(Math.sin(progress * Math.PI)) * 80;
    position.z *= 0.4;

    return position;
}

// Shape nodes to form angry (sharp angles)
private shapeAngry(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Jagged lightning-like
    position.x = Math.cos(progress * Math.PI * 8) * 100;
    position.y = Math.sin(progress * Math.PI * 8) * 100;
    position.z *= 0.5;

    return position;
}

// Shape nodes to form love (heart shape)
private shapeLove(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Parametric heart curve
    const t = progress * Math.PI * 2;
    position.x = 16 * Math.pow(Math.sin(t), 3) * 5;
    position.y = (13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t)) * 5;
    position.z *= 0.3;

    return position;
}

// Shape nodes to form fear (scattered erratic)
private shapeFear(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Erratic scattering
    position.x += Math.random() * 100 - 50;
    position.y += Math.random() * 100 - 50;
    position.z += Math.random() * 100 - 50;

    return position;
}

// Shape nodes to form joy (bouncy waves)
private shapeJoy(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Bouncy sine waves
    position.x *= 1.2;
    position.y = Math.sin(progress * Math.PI * 6) * 80;
    position.z *= 0.5;

    return position;
}

// Shape nodes to form peace (smooth circles)
private shapePeace(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Concentric circles
    const radius = Math.sqrt(progress) * 150;
    const angle = progress * Math.PI * 4;
    position.x = Math.cos(angle) * radius;
    position.y = Math.sin(angle) * radius;
    position.z *= 0.2;

    return position;
}

// Shape nodes to form intelligence (layered spheres)
private shapeIntelligence(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Layered concentric spheres
    const layers = 4;
    const layerIndex = Math.floor(progress * layers);
    const radius = (layerIndex + 1) * 50;
    position.x = Math.cos(progress * Math.PI * 2) * radius;
    position.y = Math.sin(progress * Math.PI * 2) * radius;
    position.z *= 0.5;

    return position;
}

// Shape nodes to form creativity (random bursts)
private shapeCreativity(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Burst patterns
    position.x += Math.pow(progress, 2) * 100 * (Math.random() - 0.5);
    position.y += Math.pow(progress, 2) * 100 * (Math.random() - 0.5);
    position.z += Math.pow(progress, 2) * 100 * (Math.random() - 0.5);

    return position;
}

// Shape nodes to form knowledge (book-like layers)
private shapeKnowledge(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Stacked layers like pages
    position.x *= 1.0;
    position.y *= 0.8;
    position.z = progress * 100 - 50;

    return position;
}

// Shape nodes to form wisdom (tree of life-like)
private shapeWisdom(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    if (progress < 0.5) {
        // Trunk
        position.x *= 0.4;
        position.y *= 0.4;
        position.z = progress * 150 - 75;
    } else {
        // Branches
        const branchProgress = (progress - 0.5) / 0.5;
        const branchRadius = branchProgress * 100;
        const angle = branchProgress * Math.PI * 4;
        position.x = Math.cos(angle) * branchRadius;
        position.y = Math.sin(angle) * branchRadius;
        position.z = 75 + (1 - branchProgress) * 100;
    }

    return position;
}

// Shape nodes to form energy (wave particles)
private shapeEnergy(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Oscillating waves
    position.x = Math.cos(progress * Math.PI * 10) * 120;
    position.y = Math.sin(progress * Math.PI * 10) * 120;
    position.z = Math.sin(progress * Math.PI * 5) * 60;

    return position;
}

// Shape nodes to form time (hourglass)
private shapeTime(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Hourglass funnel
    const radius = Math.abs(0.5 - progress) * 100 + 20;
    const angle = progress * Math.PI * 8;
    position.x = Math.cos(angle) * radius;
    position.y = Math.sin(angle) * radius;
    position.z = (progress - 0.5) * 200;

    return position;
}

// Shape nodes to form space (galactic swirl)
private shapeSpace(position: THREE.Vector3, index: number, totalNodes: number): THREE.Vector3 {
    const progress = index / totalNodes;

    // Swirling nebula
    const spiralArms = 4;
    const armIndex = Math.floor(progress * spiralArms);
    const armProgress = (progress * spiralArms) % 1;
    const radius = armProgress * 250;
    const angle = armProgress * Math.PI * 5 + (armIndex * Math.PI * 2 / spiralArms);

    position.x = Math.cos(angle) * radius;
    position.y = Math.sin(angle) * radius;
    position.z = Math.cos(progress * Math.PI * 3) * 80;

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
