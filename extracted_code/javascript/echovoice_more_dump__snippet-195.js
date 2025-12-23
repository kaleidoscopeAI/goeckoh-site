async function crystalIngestWeb(url) {
    const data = await fetchWebData(url);
    crystal.core.ingest({ text: data.content });
