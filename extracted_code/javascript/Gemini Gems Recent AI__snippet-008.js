try {
  const response = await axios.post('/api/similarity', { smiles1, smiles2 });
  const data = await response.json();
  setSimilarity(data.similarity);
} catch (error) {
  console.error('Error:', error);
}
