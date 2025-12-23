import React, { useState } from 'react';

const App = () => {
  const [smiles1, setSmiles1] = useState('');
  const [smiles2, setSmiles2] = useState('');
  const [similarity, setSimilarity] = useState(null);

  const calculateSimilarity = async () => {
    try {
      const response = await fetch('/api/similarity', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ smiles1, smiles2 }),
      });
      const data = await response.json();
      setSimilarity(data.similarity);
    } catch (error) {
      console.error('Error calculating similarity:', error);
    }
  };

  return (
    <div>
      <h1>Molecular Similarity</h1>
      <input type="text" placeholder="SMILES 1" value={smiles1} onChange={(e) => setSmiles1(e.target.value)} />
      <input type="text" placeholder="SMILES 2" value={smiles2} onChange={(e) => setSmiles2(e.target.value)} />
      <button onClick={calculateSimilarity}>Calculate Similarity</button>
      {similarity!== null && <p>Similarity: {similarity}</p>}
    </div>
  );
