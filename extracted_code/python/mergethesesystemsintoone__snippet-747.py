def __init__(self, dimensions: int = 12, banks_per_dimension: int = 3):
    self.dimensions = dimensions
    self.banks_per_dimension = banks_per_dimension
    self.quantum_states = np.zeros((dimensions, banks_per_dimension), dtype=np.complex128)
    self.entanglement_graph = nx.Graph()
    self.resonance_matrix = np.zeros((dimensions, dimensions), dtype=np.complex128)
    self.memory_graph = nx.DiGraph()
    self.absorbed_data = []  # Stores data absorbed from recycled nodes
    self._initialize_quantum_states()
    self.speculative_threshold = 0.6

    # Load the shared library for C backend operations
    self.c_lib = ctypes.CDLL("./c_backend/kaleidoscope_core.so")
    self._setup_c_functions()

def _setup_c_functions(self):
    """Defines argument and return types for C functions."""
    # Define argument and return types for complex_fft_fftw
    self.c_lib.complex_fft_fftw.argtypes = [np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"), ctypes.c_int]
    self.c_lib.complex_fft_fftw.restype = ctypes.c_int

    # Define argument and return types for complex_array_add
    self.c_lib.complex_array_add.argtypes = [
        np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"),
        ctypes.c_int
    ]
    self.c_lib.complex_array_add.restype = ctypes.c_int

    # Define argument and return types for complex_array_multiply
    self.c_lib.complex_array_multiply.argtypes = [
        np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"),
        ctypes.c_int
    ]
    self.c_lib.complex_array_multiply.restype = ctypes.c_int

    # Define argument and return types for hadamard_transform
    self.c_lib.hadamard_transform.argtypes = [np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS")]
    self.c_lib.hadamard_transform.restype = ctypes.c_int

    # Define argument and return types for phase_rotation
    self.c_lib.phase_rotation.argtypes = [np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS"), ctypes.c_double]
    self.c_lib.phase_rotation.restype = ctypes.c_int

    # Define argument and return types for calculate_density_matrix
    self.c_lib.calculate_density_matrix.argtypes = [
        ctypes.c_double,  # Assuming state is passed as a complex number
        np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS") # Expecting a 2x2 matrix
    ]
    self.c_lib.calculate_density_matrix.restype = ctypes.c_int

    # Define argument and return types for calculate_purity
    self.c_lib.calculate_purity.argtypes = [np.ctypeslib.ndpointer(np.complex128, flags="C_CONTIGUOUS")] # Expecting a 2x2 matrix
    self.c_lib.calculate_purity.restype = ctypes.c_double

    # Define argument and return types for measure_state
    self.c_lib.measure_state.argtypes = [ctypes.c_double] # Assuming state is passed as a complex number
    self.c_lib.measure_state.restype = ctypes.c_int

def _initialize_quantum_states(self):
    """Initialize quantum states with superposition and entanglement."""
    for dim in range(self.dimensions):
        for bank in range(self.banks_per_dimension):
            # Create superposition of states within each bank
            initial_state = np.random.rand(2) + 1j * np.random.rand(2)
            initial_state /= np.linalg.norm(initial_state)
            self.quantum_states[dim, bank] = initial_state[0]

        # Add nodes to entanglement graph
        self.entanglement_graph.add_node(dim)

    # Create initial entanglement between dimensions
    for dim1 in range(self.dimensions):
        for dim2 in range(dim1 + 1, self.dimensions):
            if np.random.rand() < 0.3:  # 30% chance of initial entanglement
                self.entanglement_graph.add_edge(dim1, dim2)
                self.resonance_matrix[dim1, dim2] = self.resonance_matrix[dim2, dim1] = np.random.rand() + 1j * np.random.rand()

def process_data(self, data: np.ndarray) -> np.ndarray:
    """Process input data through quantum transformations and resonance."""
    transformed_data = self._apply_quantum_transformations(data)
    resonance_effects = self._calculate_resonance(transformed_data)
    return resonance_effects

def _apply_quantum_transformations(self, data: np.ndarray) -> np.ndarray:
    """Apply quantum transformations to input data using matrix operations."""
    transformed_data = np.zeros_like(data, dtype=np.complex128)
    for dim in range(self.dimensions):
        for bank in range(self.banks_per_dimension):
            # Apply Hadamard gate for superposition
            hadamard_transformed = self._hadamard_transform(self.quantum_states[dim, bank])

            # Apply phase rotation based on entanglement
            phase_transformed = self._apply_phase_rotation(hadamard_transformed, dim)

            # Update quantum state
            self.quantum_states[dim, bank] = phase_transformed

            # Map input data to quantum state
            transformed_data[dim] += phase_transformed * data[dim, bank % data.shape[1]]

    # Apply FFT using the C backend
    for i in range(transformed_data.shape[0]):
        result = self.c_lib.complex_fft_fftw(transformed_data[i], transformed_data.shape[1])
        if result != 0:
            raise RuntimeError(f"Error in complex_fft_fftw for dimension {i}, error code: {result}")

    return transformed_data

def apply_fft_to_row(self, row: np.ndarray) -> np.ndarray:
    """Applies FFT to a single row of data."""
    n = len(row)
    if n <= 1:
        return row

    # Allocate memory for FFTW input and output arrays
    in_array = np.zeros(n, dtype=np.complex128)
    out_array = np.zeros(n, dtype=np.complex128)

    # Copy data to input array
    in_array[:] = row

    # Create an FFTW plan for a 1D complex-to-complex transform
    plan = self._create_fftw_plan(n, in_array, out_array)
    if plan is None:
        raise RuntimeError("Failed to create FFTW plan")

    # Execute the FFTW plan
    self._execute_fftw_plan(plan)

    # Copy data from FFTW output array to a new NumPy array
    result = np.zeros(n, dtype=np.complex128)
    result[:] = out_array

    # Clean up FFTW resources
    self._destroy_fftw_plan(plan)

    return result

def _create_fftw_plan(self, n: int, in_array: np.ndarray, out_array: np.ndarray):
    """Creates an FFTW plan for 1D complex-to-complex transform."""
    # Ensure the arrays are C-contiguous and of the correct type
    in_array = np.ascontiguousarray(in_array, dtype=np.complex128)
    out_array = np.ascontiguousarray(out_array, dtype=np.complex128)

    # Create the plan
    plan = self.c_lib.create_fftw_plan(
        ctypes.c_int(n),
        np.ctypeslib.as_ctypes(in_array),
        np.ctypeslib.as_ctypes(out_array),
        ctypes.c_int(1),  # FFTW_FORWARD
        ctypes.c_int(64)   # FFTW_ESTIMATE
    )
    if not plan:
        raise RuntimeError("Failed to create FFTW plan")

    return plan

def _execute_fftw_plan(self, plan):
    """Executes the FFTW plan."""
    self.c_lib.execute_fftw_plan(plan)

def _destroy_fftw_plan(self, plan):
    """Destroys the FFTW plan."""
    self.c_lib.destroy_fftw_plan(plan)

def _hadamard_transform(self, state: complex) -> complex:
    """Apply Hadamard transformation to a given quantum state."""
    state_np = np.array([state], dtype=np.complex128)
    result = self.c_lib.hadamard_transform(state_np)
    if result != 0:
        raise RuntimeError(f"Error in hadamard_transform, error code: {result}")
    return state_np[0]

def _apply_phase_rotation(self, state: complex, dim: int) -> complex:
    """Apply phase rotation based on entanglement with other dimensions."""
    phase_shift = 0
    for other_dim in self.entanglement_graph.neighbors(dim):
        entanglement_strength = np.abs(self.resonance_matrix[dim, other_dim])
        phase_shift += entanglement_strength * np.angle(self.quantum_states[other_dim, 0])

    # Convert the complex scalar to a numpy array for compatibility with C function
    state_np = np.array([state], dtype=np.complex128)

    # Call the C function
    result = self.c_lib.phase_rotation(state_np, phase_shift)

    if result != 0:
        raise RuntimeError(f"Error in phase_rotation for dimension {dim}, error code: {result}")

    # Return the transformed state as a complex scalar
    return state_np[0]

def _calculate_resonance(self, data: np.ndarray) -> np.ndarray:
    """Calculate resonance effects based on quantum states and entanglement."""
    resonance_effects = np.zeros_like(data, dtype=np.complex128)
    for dim in range(self.dimensions):
        for bank in range(self.banks_per_dimension):
            # Calculate resonance based on entanglement strength and phase differences
            resonance_strength = 0
            for other_dim in self.entanglement_graph.neighbors(dim):
                resonance_strength += np.abs(self.resonance_matrix[dim, other_dim]) * np.cos(
                    np.angle(self.quantum_states[dim, bank]) - np.angle(self.quantum_states[other_dim, 0])
                )
            resonance_effects[dim] += resonance_strength * data[dim]

    return resonance_effects

def update_entanglement(self, threshold: float = 0.5):
    """Update entanglement graph based on resonance matrix strengths."""
    for dim1 in range(self.dimensions):
        for dim2 in range(dim1 + 1, self.dimensions):
            resonance_strength = np.abs(self.resonance_matrix[dim1, dim2])
            if resonance_strength > threshold:
                if not self.entanglement_graph.has_edge(dim1, dim2):
                    self.entanglement_graph.add_edge(dim1, dim2)
            else:
                if self.entanglement_graph.has_edge(dim1, dim2):
                    self.entanglement_graph.remove_edge(dim1, dim2)

def get_quantum_state_metrics(self) -> dict:
    """Calculate and return metrics related to the current quantum states."""
    metrics = {
        'entanglement_entropy': self._calculate_entanglement_entropy(),
        'average_resonance': np.mean(np.abs(self.resonance_matrix)),
        'state_purity': self._calculate_average_state_purity()
    }
    return metrics

def _calculate_entanglement_entropy(self) -> float:
    """Calculate the average entanglement entropy across all dimensions."""
    entropies = []
    for dim in range(self.dimensions):
        neighbors = list(self.entanglement_graph.neighbors(dim))
        if neighbors:
            subgraph = self.entanglement_graph.subgraph(neighbors + [dim])

            # Calculate the adjacency matrix of the subgraph
            adjacency_matrix = nx.adjacency_matrix(subgraph).todense()

            # Calculate the degree matrix of the subgraph
            degree_matrix = np.diag(np.sum(adjacency_matrix, axis=1))

            # Calculate the Laplacian matrix
            laplacian_matrix = degree_matrix - adjacency_matrix

            # Calculate eigenvalues of the Laplacian
            eigenvalues = spl.eigvals(laplacian_matrix)

            # Normalize eigenvalues to represent probabilities
            probabilities = np.real(eigenvalues) / np.sum(np.real(eigenvalues))

            # Calculate entropy for the dimension
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Adding a small value to avoid log(0)
            entropies.append(entropy)
    return np.mean(entropies) if entropies else 0

def _calculate_average_state_purity(self) -> float:
    """Calculate the average purity of the quantum states across all dimensions."""
    purities = []
    for dim in range(self.dimensions):
        for bank in range(self.banks_per_dimension):
            # Calculate density matrix for the state
            state = self.quantum_states[dim, bank]
            density_matrix = np.outer([state, 1-state], [np.conj(state), np.conj(1-state)])

            # Calculate purity (Tr(ρ²))
            purity = np.real(np.trace(np.dot(density_matrix, density_matrix)))
            purities.append(purity)
    return np.mean(purities) if purities else 0

def calculate_density_matrix(self, state: complex) -> np.ndarray:
    """Calculate the density matrix for a given quantum state using the C backend."""
    density_matrix = np.zeros((2, 2), dtype=np.complex128)
    # Convert state to a NumPy array for C function compatibility
    state_np = np.array([state], dtype=np.complex128)
    result = self.c_lib.calculate_density_matrix(state_np[0], density_matrix)
    if result != 0:
        raise RuntimeError(f"Error in calculate_density_matrix, error code: {result}")
    return density_matrix

def calculate_purity(self, state: complex) -> float:
    """Calculate the purity of a given quantum state using the C backend."""
    density_matrix = self.calculate_density_matrix(state)
    result = self.c_lib.calculate_purity(density_matrix)
    if result < 0:  # Assuming negative values indicate an error
        raise RuntimeError(f"Error in calculate_purity, error code: {result}")
    return result

def measure_state(self, state: complex) -> int:
    """Simulate the measurement of a quantum state using the C backend."""
    # Convert state to a NumPy array for C function compatibility
    state_np = np.array([state], dtype=np.complex128)
    result = self.c_lib.measure_state(state_np[0])
    if result < 0: # Assuming negative values indicate an error
        raise RuntimeError(f"Error in measure_state, error code: {result}")
    return result

def add_complex_arrays(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Add two complex arrays element-wise using the C backend."""
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape for element


    result = np.zeros_like(arr1)
    return_code = self.c_lib.complex_array_add(arr1, arr2, result, arr1.size)
    if return_code != 0:
        raise RuntimeError(f"Error in complex_array_add, error code: {return_code}")
    return result

def multiply_complex_arrays(self, arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Multiply two complex arrays element-wise using the C backend."""
    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have the same shape for element-wise multiplication.")
    result = np.zeros_like(arr1)
    return_code = self.c_lib.complex_array_multiply(arr1, arr2, result, arr1.size)
    if return_code != 0:
        raise RuntimeError(f"Error in complex_array_multiply, error code: {return_code}")
    return result

def get_state(self) -> dict:
    """
    Returns the current state of the Quantum Engine.
    """
    return {
        'dimensions': self.dimensions,
        'banks_per_dimension': self.banks_per_dimension,
        'quantum_states': self.quantum_states.tolist(),  # Convert NumPy array to list for serialization
        'entanglement_graph': nx.to_dict_of_lists(self.entanglement_graph),  # Convert graph to a serializable format
        'resonance_matrix': self.resonance_matrix.tolist()  # Convert NumPy array to list
    }

def absorb_data(self, node_memory: List[Dict]):
    """
    Absorbs data from a node's memory.
    Processes and integrates data from recycled nodes.

    Args:
        node_memory: List of data entries from the node's memory.
    """
    print(f"Kaleidoscope Engine absorbing data from node. Number of entries: {len(node_memory)}")

    for data_entry in node_memory:
        # Store the raw data for potential use
        self.absorbed_data.append(data_entry)

        # Process the data to generate insights
        insights = self.process_data_for_insights(data_entry)

        # Merge the new insights into the memory graph
        self.merge_insights(insights)

def process_data_for_insights(self, data_entry: Dict) -> List[Dict]:
    """
    Processes a data entry to generate insights.
    This is a placeholder for the actual insight generation logic.
    """
    insights = []
    data = data_entry.get('data')
    metadata = data_entry.get('metadata')

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, str):
                # Example: Simple text processing
                insights.append({
                    'id': str(uuid.uuid4()),
                    'type': 'text_insight',
                    'source': key,
                    'content': value,
                    'timestamp': metadata.get('timestamp', str(np.datetime64('now'))),
                    'entities': self._extract_entities(value)  # Placeholder for entity extraction
                })
            elif isinstance(value, (int, float)):
                # Example: Numerical data processing
                insights.append({
                    'id': str(uuid.uuid4()),
                    'type': 'numerical_insight',
                    'source': key,
                    'value': value,
                    'timestamp': metadata.get('timestamp', str(np.datetime64('now')))
                })
            # Add more types as needed
    # Placeholder for more complex data processing and insight generation
    return insights

def _extract_entities(self, text: str) -> List[str]:
    """
    Placeholder for extracting entities from text.
    """
    # Implement entity extraction logic here (e.g., using NLP techniques)
    return []    

def get_insights(self, since_timestamp: Optional[str] = None) -> List[Dict]:
    """
    Retrieves validated insights from the engine's memory graph.

    Args:
        since_timestamp (Optional[str]): Optional timestamp to filter insights.
                                         Only retrieves insights created after this timestamp.

    Returns:
        List[Dict]: A list of dictionaries, each representing a validated insight.
    """
    insights = []
    for node_id, node_data in self.memory_graph.nodes(data=True):
        insight = node_data.get('attributes', {})
        insight['id'] = node_id  # Ensure the insight ID is included

        # Check if a timestamp filter is provided
        if since_timestamp is not None:
            insight_timestamp = insight.get('timestamp')
            if insight_timestamp:
                try:
                    # Parse the timestamp string into a datetime object
                    insight_timestamp = datetime.fromisoformat(insight_timestamp)
                    # Parse the filter timestamp into a datetime object
                    filter_timestamp = datetime.fromisoformat(since_timestamp)
                    if insight_timestamp <= filter_timestamp:
                        continue  # Skip insights older than the filter timestamp
                except ValueError:
                    print(f"Error parsing timestamp for insight {node_id}. Skipping timestamp filter.")

        insights.append(insight)

    return insights

def merge_insights(self, insights: List[Dict]):
    """
    Merges new insights into the memory graph, connecting related insights.
    """
    for insight in insights:
        insight_id = insight['id']
        self.memory_graph.add_node(insight_id, attributes=insight)

        # Connect to related insights based on shared entities or dimensions
        for other_insight_id, other_insight_data in self.memory_graph.nodes(data=True):
            if other_insight_id != insight_id:
                other_insight_attrs = other_insight_data.get('attributes', {})
                if self._insights_relate(insight, other_insight_attrs):
                    self.memory_graph.add_edge(insight_id, other_insight_id)

def _insights_relate(self, insight1: Dict, insight2: Dict) -> bool:
    """
    Determines if two insights are related based on their attributes.

    Args:
        insight1 (Dict): The first insight.
        insight2 (Dict): The second insight.

    Returns:
        bool: True if the insights are related, False otherwise.
    """
    # Check for direct entity overlap
    if self._insights_share_entities(insight1, insight2):
        return True

    # Check for temporal proximity
    if self._insights_are_temporally_close(insight1, insight2):
        return True

    # Check for indirect relationships via graph connectivity
    if self._insights_indirectly_related(insight1, insight2):
        return True

    return False

def _insights_share_entities(self, insight1: Dict, insight2: Dict) -> bool:
    """
    Checks if two insights share common entities.
    """
    entities1 = set(insight1.get('entities', []))
    entities2 = set(insight2.get('entities', []))
    return bool(entities1.intersection(entities2))

def _insights_are_temporally_close(self, insight1: Dict, insight2: Dict, time_threshold: int = 60) -> bool:
    """
    Checks if two insights are close in time.
    """
    try:
        time1 = datetime.fromisoformat(insight1['timestamp'])
        time2 = datetime.fromisoformat(insight2['timestamp'])
        return abs((time1 - time2).total_seconds()) < time_threshold
    except (KeyError, ValueError):
        return False

def _insights_indirectly_related(self, insight1: Dict, insight2: Dict) -> bool:
    """
    Checks if two insights are indirectly related via the memory graph.
    """
    if not self.memory_graph.has_node(insight1['id']) or not self.memory_graph.has_node(insight2['id']):
        return False
    try:
        return nx.has_path(self.memory_graph, insight1['id'], insight2['id'])
    except nx.NodeNotFound:
        return False

def collaborate_with_perspective_engine(self, perspective_engine):
    """
    Collaborates with the PerspectiveEngine to create CollectiveNode instances.
    """
    # Get speculative insights from the PerspectiveEngine
    speculative_insights = perspective_engine.get_speculative_insights()

    # Combine validated and speculative insights
    combined_insights = self.merge_insights_and_perspectives(speculative_insights)

    # Create CollectiveNode instances based on the combined insights
    collective_node = self._create_collective_node(combined_insights)
    return collective_node

def _create_collective_node(self, insights: List[Dict]) -> CollectiveNode:
    """
    Creates a CollectiveNode instance from a set of insights.
    """
    collective_node_id = str(uuid.uuid4())
    collective_node = CollectiveNode(collective_node_id, self.dimensions)
    collective_node.update_insights(insights)
    return collective_node

def merge_insights_and_perspectives(self, speculative_insights):
    """
    Merges validated insights with speculative insights from the PerspectiveEngine.
    Prioritizes validated insights and uses speculative insights to fill gaps or 
    provide alternative viewpoints.
    """
    merged_insights = []

    # Start with validated insights from the memory graph
    validated_insights = self.get_insights()
    merged_insights.extend(validated_insights)

    # Add speculative insights, resolving conflicts based on confidence and timestamps
    for speculative_insight in speculative_insights:
        is_duplicate = False
        for validated_insight in validated_insights:
            # Check for duplicates based on insight type and source
            if speculative_insight['type'] == validated_insight['type'] and \
               speculative_insight['source'] == validated_insight['source']:
                is_duplicate = True

                # Resolve conflict: higher confidence or more recent insight wins
                if speculative_insight['confidence'] > validated_insight['confidence'] or \
                   (speculative_insight['confidence'] == validated_insight['confidence'] and 
                    speculative_insight['timestamp'] > validated_insight['timestamp']):

                    # Replace the validated insight with the speculative one
                    merged_insights.remove(validated_insight)
                    merged_insights.append(speculative_insight)
                break

        if not is_duplicate:
            merged_insights.append(speculative_insight)

    return merged_insights

# ... (other methods remain the same)










