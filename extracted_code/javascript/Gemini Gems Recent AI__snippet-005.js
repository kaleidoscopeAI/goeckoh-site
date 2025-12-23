try {
  const response = await axios.post('/api/similarity', { smiles1, smiles2 });
  const taskId = response.data.task_id;

  // Poll for the result (not ideal, but a simple example)
  const checkResult = async () => {
    const resultResponse = await axios.get(`/api/similarity_result/${taskId}`);
    if (resultResponse.data.status === 'SUCCESS') {
      setSimilarityResult(resultResponse.data.result);
    } else {
      setTimeout(checkResult, 1000); // Check again after 1 second
    }
  };

  checkResult();
} catch (error) {
  console.error("Error searching for similarity:", error);
}
