import React, { useState, useEffect, useRef } from 'react';
import * as THREE from 'three';
import axios from 'axios';

const App = () => {
    const mountRef = useRef(null);
    const [smiles1, setSmiles1] = useState('');
    const [smiles2, setSmiles2] = useState('');
    const [similarity, setSimilarity] = useState(null);
    const [features1, setFeatures1] = useState(null); // Store features for molecule 1
    const [features2, setFeatures2] = useState(null); // Store features for molecule 2

    useEffect(() => {
        //... (Three.js scene setup - same as previous example)

        const animate = function () {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        };

        animate();

        return () => {
            mountRef.current.removeChild(renderer.domElement);
        };
    },);

    const calculateSimilarity = async () => {
        try {
            const response = await axios.post('/api/similarity', { smiles1, smiles2 });
            const data = response.data;
            setSimilarity(data.similarity);

            // Fetch and store features for visualization
            const featuresResponse1 = await axios.post('/api/features', { smiles: smiles1 });
            const featuresResponse2 = await axios.post('/api/features', { smiles: smiles2 });
            setFeatures1(featuresResponse1.data.features);
            setFeatures2(featuresResponse2.data.features);

            // Now you have features1 and features2 to position objects in 3D
            console.log("Features 1:", features1);
            console.log("Features 2:", features2);
        } catch (error) {
            console.error('Error:', error);
        }
    };

    return (
        <div>
            {/*... input fields and buttons... */}
            <div ref={mountRef} style={{ height: '600px' }} /> {/* 3D visualization */}
            {/*... similarity results... */}
        </div>
    );
