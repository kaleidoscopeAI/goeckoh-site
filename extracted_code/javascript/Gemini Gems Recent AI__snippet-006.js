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
