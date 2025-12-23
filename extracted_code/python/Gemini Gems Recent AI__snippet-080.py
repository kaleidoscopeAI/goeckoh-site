import axios from 'axios';

const api = {
  getMolecules: () => axios.get('/api/molecules'),
  getSimilarity: (smiles1, smiles2) => axios.post('/api/similarity', { smiles1, smiles2 }),
  getSimilarityResult: (taskId) => axios.get(`/api/similarity_result/${taskId}`),
  //... other API calls
