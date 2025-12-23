def __init__(self, node_id: Optional[str] = None, energy: float = 100.0, stress: float = 0.5,

             position: Optional[np.ndarray] = None, parent_cube=None, smiles: Optional[str] = None):

    self.node_id = node_id if node_id else str(uuid.uuid4())

    self.energy = energy

    self.stress = stress

    self.position = position if position is not None else np.random.rand(3) * 2

    self.E_MAX = 200.0

    self.parent_cube = parent_cube

    self.smiles = smiles if smiles else "CC(=O)OC1=CC=CC=C1C(=O)O"  # Default: aspirin

    self.mol = None

    self.chem_properties = {}

    if Chem and smiles:

        try:

            self.mol = Chem.MolFromSmiles(smiles)

            if self.mol:

                self.chem_properties = {

                    "mol_weight": Descriptors.MolWt(self.mol),

                    "logp": Descriptors.MolLogP(self.mol),

                    "h_bond_donors": Descriptors.NumHDonors(self.mol),

                    "h_bond_acceptors": Descriptors.NumHAcceptors(self.mol)

                }

                logging.debug(f"Node {self.node_id}: Initialized molecule with SMILES {smiles}")

            else:

                logging.warning(f"Invalid SMILES for node {self.node_id}: {smiles}")

        except Exception as e:

            logging.error(f"Node {self.node_id} SMILES parsing failed: {e}")


def update_state(self, incoming_stress: float, energy_cost: float):

    try:

        self.energy = max(0, min(self.E_MAX, self.energy - energy_cost))

        self.stress = max(0.0, min(1.0, self.stress + incoming_stress - (self.energy / self.E_MAX) * 0.1))

        logging.debug(f"Node {self.node_id}: Energy={self.energy:.2f}, Stress={self.stress:.2f}")

    except Exception as e:

        logging.error(f"Node {self.node_id} state update failed: {e}")


def process_insight(self, insight: Insight):

    try:

        if insight.type == "threat":

            self.stress = min(1.0, self.stress + insight.confidence * 0.1)

            self.energy = max(0.0, self.energy - insight.confidence * 5)

        elif insight.type == "resource":

            self.energy = min(self.E_MAX, self.energy + insight.confidence * 10)

            self.stress = max(0.0, self.stress - insight.confidence * 0.05)

        elif insight.type == "molecular":

            self.stress = max(0.0, self.stress - insight.data.get("similarity", 0) * 0.1)

            self.energy = min(self.E_MAX, self.energy + insight.data.get("similarity", 0) * 5)

        logging.debug(f"Node {self.node_id} processed insight {insight.insight_id}: E={self.energy:.2f}, S={self.stress:.2f}")

    except Exception as e:

        logging.error(f"Node {self.node_id} insight processing failed: {e}")


def replicate(self, topo_signal: float, confidence: float) -> Optional['EmotionalNode']:

    try:

        if random.random() < 0.1 * (1.0 - self.stress) * confidence * (1.0 + topo_signal):

            new_smiles = self.smiles  # Simplified: inherit SMILES

            if Chem and self.mol:

                # Mutate SMILES slightly (placeholder for molecular mutation)

                new_smiles = self.smiles + "C" if random.random() < 0.5 else self.smiles[:-1]

                try:

                    Chem.MolFromSmiles(new_smiles)  # Validate new SMILES

                except:

                    new_smiles = self.smiles  # Fallback to original

            new_node = EmotionalNode(

                energy=self.energy * 0.8,

                stress=self.stress * 0.8,

                position=self.position + np.random.randn(3) * 0.1,

                parent_cube=self.parent_cube,

                smiles=new_smiles

            )

            logging.info(f"Node {self.node_id} replicated into {new_node.node_id} with SMILES {new_smiles}")

            return new_node

        return None

    except Exception as e:

        logging.error(f"Node {self.node_id} replication failed: {e}")

        return None


def compute_similarity(self, other_node: 'EmotionalNode') -> float:

    if not Chem or not self.mol or not other_node.mol:

        return 0.0

    try:

        fp1 = AllChem.GetMorganFingerprintAsBitVect(self.mol, 2, nBits=2048)

        fp2 = AllChem.GetMorganFingerprintAsBitVect(other_node.mol, 2, nBits=2048)

        similarity = Chem.DataStructs.TanimotoSimilarity(fp1, fp2)

        return similarity

    except Exception as e:

        logging.error(f"Similarity computation failed for node {self.node_id}: {e}")

        return 0.0


