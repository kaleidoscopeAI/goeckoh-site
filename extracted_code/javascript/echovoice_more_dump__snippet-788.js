async function handleLoadW(id: string) {
const resp = await axios.get(`http://localhost:4201/w/get/${id}`);
