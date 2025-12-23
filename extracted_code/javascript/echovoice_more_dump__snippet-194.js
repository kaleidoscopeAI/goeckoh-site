async function crystalIngestSMILES(smiles) {
    const parser = new SMILESParser();
    const atoms = parser.parse(smiles);
    crystal.core.ingest({ atoms });
