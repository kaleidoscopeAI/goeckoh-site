class QSINNode:
    """Implementation of a QSIN node"""
    
    def __init__(self, config: QSINConfig):
        self.config = config
        self.node_id = config.node_id
        self.node_name = config.node_name
        
        # Initialize state
        self.state = NodeState(node_id=self.node_id, energy=config.initial_energy)
        
        # Quantum components
        self.quantum_state = QuantumState(dimension=config.dimension)
        self.hamiltonian_gen = HamiltonianGenerator(dimension=config.dimension)
        self.observable_gen = ObservableGenerator(dimension=config.dimension)
        self.swarm_operator = SwarmOperator(dimension=config.dimension)
        
        # Network components
        self.server_mode = config.server_mode
        self.server = None
        self.client_connection = None
        self.connected = False
        self.connections = {}  # node_id -> websocket
        
        # Networking data
        self.network_graph = nx.Graph()
        self.node_mapping = {}  # node_id -> index
        self.discovered_nodes = {}  # node_id -> node_info
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Shutdown signal
        self.shutdown_event = asyncio.Event()
        
        # Tasks
        self.tasks = set()
        
        # Operation buffer
        self.operation_buffer = []
        
        # For node replication
        self.network_discovery = NetworkDiscovery(config)
        self.ssh_deployer = SSHDeployer(config)
        
        # Initialize logger
        self.logger = logging.getLogger(f"qsin-node-{self.node_id[:8]}")
    
    async def start(self) -> None:
        """Start the node"""
        self.logger.info(f"Starting QSIN node: {self.node_name} ({self.node_id})")
        
        # Initialize network graph
        self.network_graph.add_node(
            self.node_id, 
            name=self.node_name, 
            type="self",
            energy=self.state.energy
        )
        
        # Start server or client based on mode
        if self.server_mode:
            await self._start_server()
        else:
            await self._start_client()
        
        # Start background tasks
        self.tasks.add(asyncio.create_task(self._quantum_evolution_loop()))
        self.tasks.add(asyncio.create_task(self._energy_processing_loop()))
        
        if self.config.discovery_enabled:
            self.tasks.add(asyncio.create_task(self._discovery_loop()))
        
        self.state.status = "active"
        self.logger.info(f"Node {self.node_name} started successfully")
    
    async def stop(self) -> None:
        """Stop the node"""
        self.logger.info(f"Stopping node {self.node_name}")
        
        # Signal shutdown
        self.shutdown_event.set()
        
        # Cancel all tasks
        for task in self.tasks:
            task.cancel()
        
        # Close connections
        if self.server_mode and self.server:
            self.server.close()
            await self.server.wait_closed()
        elif self.client_connection:
            await self.client_connection.close()
        
        # Close all peer connections
        for conn in self.connections.values():
            await conn.close()
        
        self.state.status = "terminated"
        self.logger.info(f"Node {self.node_name} stopped")
    
    async def _start_server(self) -> None:
        """Start in server mode (listen for connections)"""
        self.logger.info(f"Starting QSIN server on {self.config.host}:{self.config.port}")
        
        try:
            # Start WebSocket server
            self.server = await websockets.serve(
                self._handle_connection,
                self.config.host,
                self.config.port
            )
            
            self.tasks.add(asyncio.create_task(self._server_maintenance_loop()))
            
            self.logger.info(f"QSIN server started on port {self.config.port}")
            
        except Exception as e:
            self.logger.error(f"Error starting server: {e}")
            raise
    
    async def _start_client(self) -> None:
        """Start in client mode (connect to server)"""
        self.logger.info(f"Starting QSIN client connecting to {self.config.server_url}")
        
        try:
            # Connect to server
            self.client_connection = await websockets.connect(self.config.server_url)
            
            # Register with server
            register_msg = NetworkMessage(
                message_type="register",
                sender_id=self.node_id,
                content={
                    "node_name": self.node_name,
                    "energy": self.state.energy,
                    "capabilities": ["compute", "storage", "entanglement"]
                }
            )
            
            await self.client_connection.send(json.dumps(register_msg.to_dict()))
            
            # Wait for registration acknowledgment
            response = await self.client_connection.recv()
            data = json.loads(response)
            
            if data.get("message_type") == "register_ack":
                self.connected = True
                self.logger.info(f"Connected to QSIN server as {self.node_name}")
                
                # Start message processing
                self.tasks.add(asyncio.create_task(self._client_message_loop()))
                self.tasks.add(asyncio.create_task(self._client_reconnect_loop()))
                
            else:
                self.logger.error(f"Registration failed: {data}")
                await self.client_connection.close()
                raise RuntimeError("Registration failed")
                
        except Exception as e:
            self.logger.error(f"Error starting client: {e}")
            raise
    
    async def _handle_connection(self, websocket, path) -> None:
        """Handle incoming connection in server mode"""
        try:
            # Receive initial message
            message = await websocket.recv()
            data = json.loads(message)
            
            # Convert to NetworkMessage
            if isinstance(data, dict) and "message_type" in data:
                msg = NetworkMessage.from_dict(data)
            else:
                # Legacy message format
                msg_type = data.get("type", "unknown")
                msg = NetworkMessage(
                    message_type=msg_type,
                    sender_id=data.get("node_id", "unknown"),
                    content=data
                )
            
            # Process based on message type
            if msg.message_type == "register":
                # Node registration
                node_id = msg.sender_id
                node_name = msg.content.get("node_name", f"Node-{node_id[:8]}")
                
                self.logger.info(f"Node registration: {node_name} ({node_id})")
                
                # Store connection
                self.connections[node_id] = websocket
                
                # Add to network graph
                self.network_graph.add_node(
                    node_id,
                    name=node_name,
                    type="client",
                    energy=msg.content.get("energy", 0.0),
                    last_seen=time.time()
                )
                
                # Acknowledge registration
                ack_msg = NetworkMessage(
                    message_type="register_ack",
                    sender_id=self.node_id,
                    receiver_id=node_id,
                    content={
                        "status": "success",
                        "server_name": self.node_name,
                        "timestamp": time.time()
                    }
                )
                
                await websocket.send(json.dumps(ack_msg.to_dict()))
                
                # Process messages from this node
                await self._process_node_messages(node_id, websocket)
                
            elif msg.message_type == "discovery_ping":
                # Network discovery ping
                source_id = msg.sender_id
                source_name = msg.content.get("source_name", f"Node-{source_id[:8]}")
                
                self.logger.debug(f"Discovery ping from {source_name} ({source_id})")
                
                # Respond with pong
                pong_msg = NetworkMessage(
                    message_type="discovery_pong",
                    sender_id=self.node_id,
                    receiver_id=source_id,
                    content={
                        "node_id": self.node_id,
                        "node_name": self.node_name,
                        "capabilities": ["compute", "storage", "entanglement"],
                        "timestamp": time.time()
                    }
                )
                
                await websocket.send(json.dumps(pong_msg.to_dict()))
                
            else:
                # Unknown message type
                self.logger.warning(f"Unknown initial message type: {msg.message_type}")
        
        except Exception as e:
            self.logger.error(f"Error handling connection: {e}")
            await websocket.close()
    
    async def _process_node_messages(self, node_id: str, websocket) -> None:
        """Process messages from a connected node"""
        try:
            async for message in websocket:
                # Parse message
                data = json.loads(message)
                
                # Convert to NetworkMessage
                if isinstance(data, dict) and "message_type" in data:
                    msg = NetworkMessage.from_dict(data)
                else:
                    # Legacy message format
                    msg_type = data.get("type", "unknown")
                    msg = NetworkMessage(
                        message_type=msg_type,
                        sender_id=node_id,
                        content=data
                    )
                
                # Handle based on message type
                response = await self._handle_message(msg)
                
                # Send response if any
                if response:
                    await websocket.send(json.dumps(response.to_dict()))
                
                # Update node in graph
                if node_id in self.network_graph:
                    self.network_graph.nodes[node_id]['last_seen'] = time.time()
                    
                    # Update energy if provided
                    if msg.message_type == "energy_update":
                        self.network_graph.nodes[node_id]['energy'] = msg.content.get("energy", 0.0)
                
        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Connection closed for node {node_id}")
        except Exception as e:
            self.logger.error(f"Error processing messages from {node_id}: {e}")
        finally:
            # Clean up connection
            if node_id in self.connections:
                del self.connections[node_id]
    
    async def _handle_message(self, msg: NetworkMessage) -> Optional[NetworkMessage]:
        """Handle a network message"""
        if msg.message_type == "ping":
            # Respond to ping
            return NetworkMessage(
                message_type="pong",
                sender_id=self.node_id,
                receiver_id=msg.sender_id,
                content={"timestamp": time.time()}
            )
        
        elif msg.message_type == "energy_update":
            # Node energy update
            node_id = msg.sender_id
            energy = msg.content.get("energy", 0.0)
            
            if node_id in self.network_graph:
                self.network_graph.nodes[node_id]["energy"] = energy
            
            return None
        
        elif msg.message_type == "quantum_state_update":
            # Quantum state update
            node_id = msg.sender_id
            
            if "state" in msg.content:
                try:
                    # Deserialize quantum state
                    remote_state = QuantumState.deserialize(msg.content["state"])
                    
                    # Process quantum state update
                    # For demonstration, we'll just calculate coherence
                    coherence = 0.0
                    if self.quantum_state and remote_state:
                        coherence = np.abs(np.vdot(self.quantum_state.state, remote_state.state))**2
                    
                    # Store coherence in graph edge
                    if node_id in self.network_graph:
                        if not self.network_graph.has_edge(self.node_id, node_id):
                            self.network_graph.add_edge(self.node_id, node_id, weight=coherence)
                        else:
                            self.network_graph[self.node_id][node_id]["weight"] = coherence
                        
                        # Update entanglement if coherence is high
                        if coherence > ENTANGLEMENT_THRESHOLD:
                            self.quantum_state.entangle_with(node_id)
                            self.state.entangled_nodes.add(node_id)
                    
                    return NetworkMessage(
                        message_type="quantum_state_ack",
                        sender_id=self.node_id,
                        receiver_id=msg.sender_id,
                        content={"coherence": coherence}
                    )
                    
                except Exception as e:
                    self.logger.error(f"Error processing quantum state update: {e}")
            
            return None
        
        elif msg.message_type == "task_request":
            # Process task request
            task_id = msg.content.get("task_id", str(uuid.uuid4()))
            task_type = msg.content.get("task_type", "optimization")
            
            self.logger.info(f"Received task request: {task_id} ({task_type})")
            
            # Process task in background
            asyncio.create_task(self._process_task(task_id, task_type, msg.sender_id))
            
            return NetworkMessage(
                message_type="task_ack",
                sender_id=self.node_id,
                receiver_id=msg.sender_id,
                content={
                    "task_id": task_id,
                    "status": "processing"
                }
            )
        
        elif msg.message_type == "task_result":
            # Process task result
            task_id = msg.content.get("task_id", "unknown")
            result = msg.content.get("result", 0.0)
            
            self.logger.info(f"Received task result from {msg.sender_id}: {task_id} = {result}")
            
            # Store result (in a real system, would have a task manager)
            # Acknowledge result
            return NetworkMessage(
                message_type="result_ack",
                sender_id=self.node_id,
                receiver_id=msg.sender_id,
                content={
                    "task_id": task_id,
                    "status": "received"
                }
            )
        
        elif msg.message_type == "entanglement_request":
            # Request for quantum entanglement
            target_id = msg.sender_id
            
            self.logger.info(f"Entanglement request from {target_id}")
            
            # Create entangled state
            success, coherence = await self._create_entanglement(target_id)
            
            if success:
                return NetworkMessage(
                    message_type="entanglement_success",
                    sender_id=self.node_id,
                    receiver_id=target_id,
                    content={
                        "coherence": coherence,
                        "timestamp": time.time()
                    }
                )
            else:
                return NetworkMessage(
                    message_type="entanglement_failure",
                    sender_id=self.node_id,
                    receiver_id=target_id,
                    content={
                        "reason": "Failed to create entanglement"
                    }
                )
        
        elif msg.message_type == "replication_request":
            # Request to replicate to new host
            target_host = msg.content.get("target_host")
            if not target_host:
                return NetworkMessage(
                    message_type="replication_failure",
                    sender_id=self.node_id,
                    receiver_id=msg.sender_id,
                    content={
                        "reason": "No target host specified"
                    }
                )
            
            # Start replication in background
            asyncio.create_task(self._replicate_to_host(target_host))
            
            return NetworkMessage(
                message_type="replication_ack",
                sender_id=self.node_id,
                receiver_id=msg.sender_id,
                content={
                    "target_host": target_host,
                    "status": "started"
                }
            )
        
        # Unknown message type
        return None
    
    async def _process_task(self, task_id: str, task_type: str, sender_id: str) -> None:
        """Process a compute task"""
        self.logger.info(f"Processing task {task_id} of type {task_type}")
        
        try:
            # Different task types
            if task_type == "optimization":
                # Use swarm optimization
                result = self.swarm_operator.solve_optimization_task(
                    task_id,
                    self.node_id,
                    self.quantum_state
                )
                
                # Convert result to message format
                result_msg = NetworkMessage(
                    message_type="task_result",
                    sender_id=self.node_id,
                    receiver_id=sender_id,
                    content={
                        "task_id": task_id,
                        "task_type": task_type,
                        "result_value": result.result_value,
                        "confidence": result.confidence,
                        "computation_time": result.computation_time,
                        "metadata": result.metadata
                    }
                )
                
                # Consume energy for task
                task_energy_cost = 10.0 + 0.1 * result.computation_time
                with self.lock:
                    self.state.energy -= task_energy_cost
                    self.state.energy = max(0.0, self.state.energy)
                    self.state.processed_tasks += 1
                
                # Send result
                if sender_id in self.connections:
                    await self.connections[sender_id].send(json.dumps(result_msg.to_dict()))
                elif self.client_connection and self.connected:
                    await self.client_connection.send(json.dumps(result_msg.to_dict()))
                
            elif task_type == "measurement":
                # Quantum measurement task
                observable_type = task_id.split("_")[0] if "_" in task_id else "random"
                
                # Get appropriate observable
                if observable_type == "energy":
                    observable = self.observable_gen.get_energy_observable()
                elif observable_type == "coherence":
                    observable = self.observable_gen.get_coherence_observable()
                else:
                    observable = self.observable_gen.get_random_observable()
                
                # Perform measurement (with collapse)
                measurement, new_state = self.quantum_state.measure_with_collapse(observable)
                
                # Send measurement result
                result_msg = NetworkMessage(
                    message_type="task_result",
                    sender_id=self.node_id,
                    receiver_id=sender_id,
                    content={
                        "task_id": task_id,
                        "task_type": task_type,
                        "result_value": float(measurement),
                        "confidence": float(new_state.fidelity),
                        "collapse_status": new_state.collapse_status.name
                    }
                )
                
                # Send result
                if sender_id in self.connections:
                    await self.connections[sender_id].send(json.dumps(result_msg.to_dict()))
                elif self.client_connection and self.connected:
                    await self.client_connection.send(json.dumps(result_msg.to_dict()))
                
                # Replace state with collapsed state
                self.quantum_state = new_state
            
            else:
                # Unknown task type
                self.logger.warning(f"Unknown task type: {task_type}")
                
                # Send error
                error_msg = NetworkMessage(
                    message_type="task_error",
                    sender_id=self.node_id,
                    receiver_id=sender_id,
                    content={
                        "task_id": task_id,
                        "error": f"Unknown task type: {task_type}"
                    }
                )
                
                # Send error
                if sender_id in self.connections:
                    await self.connections[sender_id].send(json.dumps(error_msg.to_dict()))
                elif self.client_connection and self.connected:
                    await self.client_connection.send(json.dumps(error_msg.to_dict()))
        
        except Exception as e:
            self.logger.error(f"Error processing task {task_id}: {e}")
            
            # Send error
            error_msg = NetworkMessage(
                message_type="task_error",
                sender_id=self.node_id,
                receiver_id=sender_id,
                content={
                    "task_id": task_id,
                    "error": str(e)
                }
            )
            
            # Send error
            if sender_id in self.connections:
                await self.connections[sender_id].send(json.dumps(error_msg.to_dict()))
            elif self.client_connection and self.connected:
                await self.client_connection.send(json.dumps(error_msg.to_dict()))
    
    async def _create_entanglement(self, target_id: str) -> Tuple[bool, float]:
        """Create quantum entanglement with another node"""
        self.logger.info(f"Creating entanglement with {target_id}")
        
        try:
            # Generate entanglement observable
            entanglement_obs = self.observable_gen.get_entanglement_observable(self.node_id, target_id)
            
            # Measure to create entanglement
            measurement, new_state = self.quantum_state.measure_with_collapse(entanglement_obs)
            
            # Set state to entangled
            new_state.entangle_with(target_id)
            new_state.collapse_status = WavefunctionCollapse.ENTANGLED
            
            # Replace quantum state
            self.quantum_state = new_state
            
            # Update node state
            self.state.entangled_nodes.add(target_id)
            
            # Calculate coherence (proxy for entanglement quality)
            coherence = measurement
            
            # Add to network graph
            if target_id in self.network_graph:
                if not self.network_graph.has_edge(self.node_id, target_id):
                    self.network_graph.add_edge(self.node_id, target_id, weight=coherence)
                else:
                    self.network_graph[self.node_id][target_id]["weight"] = coherence
                    self.network_graph[self.node_id][target_id]["entangled"] = True
            
            return True, coherence
            
        except Exception as e:
            self.logger.error(f"Error creating entanglement with {target_id}: {e}")
            return False, 0.0
    
    async def _quantum_evolution_loop(self) -> None:
        """Quantum state evolution background task"""
        self.logger.info("Starting quantum evolution loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Update node mapping
                self.node_mapping = {n: i for i, n in enumerate(self.network_graph.nodes())}
                
                # Create Hamiltonian
                hamiltonian = self.hamiltonian_gen.get_evolution_hamiltonian(
                    self.network_graph,
                    self.node_mapping
                )
                
                # Evolve quantum state
                self.quantum_state.evolve(hamiltonian, dt=0.1)
                
                # Apply decoherence
                self.quantum_state.apply_noise(dt=0.1)
                
                # Occasionally share state with connected nodes
                if random.random() < 0.2:  # 20% chance each cycle
                    await self._share_quantum_state()
                
                # Wait before next evolution step
                await asyncio.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in quantum evolution loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _share_quantum_state(self) -> None:
        """Share quantum state with connected nodes"""
        # Serialize quantum state
        state_data = self.quantum_state.serialize()
        
        # Create message
        state_msg = NetworkMessage(
            message_type="quantum_state_update",
            sender_id=self.node_id,
            content={
                "state": state_data
            }
        )
        
        # Convert to JSON
        state_json = json.dumps(state_msg.to_dict())
        
        # Share with all connected nodes in server mode
        if self.server_mode:
            # Send to all connections
            for node_id, conn in list(self.connections.items()):
                try:
                    await conn.send(state_json)
                except:
                    # Connection error, will be cleaned up in maintenance loop
                    pass
        
        # Share with server in client mode
        elif self.client_connection and self.connected:
            try:
                await self.client_connection.send(state_json)
            except:
                # Connection error, will attempt reconnect
                self.connected = False
    
    async def _energy_processing_loop(self) -> None:
        """Energy processing and growth background task"""
        self.logger.info("Starting energy processing loop")
        
        while not self.shutdown_event.is_set():
            try:
                with self.lock:
                    # Calculate energy gain based on quantum state
                    # Higher coherence = higher energy gain
                    base_gain = 1.0
                    coherence_factor = self.quantum_state.fidelity
                    entanglement_bonus = 0.5 * len(self.state.entangled_nodes)
                    
                    energy_gain = base_gain * coherence_factor + entanglement_bonus
                    
                    # Apply energy gain
                    self.state.energy += energy_gain
                    
                    # Check for replication
                    if self.state.energy >= self.config.replication_threshold:
                        # Start replication in background
                        asyncio.create_task(self._replicate_node())
                
                # Report energy if in client mode
                if not self.server_mode and self.client_connection and self.connected:
                    # Send energy update
                    energy_msg = NetworkMessage(
                        message_type="energy_update",
                        sender_id=self.node_id,
                        content={
                            "energy": self.state.energy,
                            "timestamp": time.time()
                        }
                    )
                    
                    try:
                        await self.client_connection.send(json.dumps(energy_msg.to_dict()))
                    except:
                        # Connection error, will attempt reconnect
                        self.connected = False
                
                # Wait before next processing
                await asyncio.sleep(5.0)
                
            except Exception as e:
                self.logger.error(f"Error in energy processing loop: {e}")
                await asyncio.sleep(5.0)
    
    async def _discovery_loop(self) -> None:
        """Network discovery background task"""
        self.logger.info("Starting network discovery loop")
        
        discovery_attempts = 0
        
        # Initial delay to allow node to stabilize
        await asyncio.sleep(10.0)
        
        while not self.shutdown_event.is_set() and discovery_attempts < self.config.max_discovery_attempts:
            try:
                # Skip if client mode and not connected
                if not self.server_mode and not self.connected:
                    await asyncio.sleep(10.0)
                    continue
                
                # Only server mode or well-connected client nodes should discover
                if self.server_mode or len(self.state.connected_nodes) > 2:
                    # Discover nodes on network
                    discovered = await self.network_discovery.discover_nodes()
                    
                    if discovered:
                        self.logger.info(f"Discovered {len(discovered)} nodes")
                        
                        # Add to discovered nodes
                        for node in discovered:
                            node_id = node["node_id"]
                            
                            # Skip self
                            if node_id == self.node_id:
                                continue
                                
                            # Store discovered node
                            self.discovered_nodes[node_id] = node
                            
                            # Try to connect
                            if self.server_mode:
                                # Connect in background
                                asyncio.create_task(self._connect_to_node(node))
                
                discovery_attempts += 1
                
                # Longer wait between discovery attempts
                await asyncio.sleep(30.0)
                
            except Exception as e:
                self.logger.error(f"Error in discovery loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _connect_to_node(self, node_info: Dict[str, Any]) -> None:
        """Connect to a discovered node"""
        node_id = node_info["node_id"]
        ip_address = node_info["ip_address"]
        port = node_info.get("port", DEFAULT_PORT)
        
        # Skip if already connected
        if node_id in self.connections or node_id in self.state.connected_nodes:
            return
        
        try:
            # Connect to node
            ws_url = f"ws://{ip_address}:{port}/qsin"
            websocket = await websockets.connect(ws_url)
            
            # Register with node
            register_msg = NetworkMessage(
                message_type="register",
                sender_id=self.node_id,
                receiver_id=node_id,
                content={
                    "node_name": self.node_name,
                    "energy": self.state.energy,
                    "capabilities": ["compute", "storage", "entanglement"]
                }
            )
            
            await websocket.send(json.dumps(register_msg.to_dict()))
            
            # Wait for registration acknowledgment
            response = await websocket.recv()
            data = json.loads(response)
            
            if data.get("message_type") == "register_ack":
                # Store connection
                self.connections[node_id] = websocket
                self.state.connected_nodes.add(node_id)
                
                # Add to network graph
                self.network_graph.add_node(
                    node_id,
                    name=node_info.get("node_name", f"Node-{node_id[:8]}"),
                    type="peer",
                    energy=0.0,
                    last_seen=time.time()
                )
                
                # Start message processing
                asyncio.create_task(self._process_node_messages(node_id, websocket))
                
                self.logger.info(f"Connected to node {node_id}")
            else:
                # Registration failed
                await websocket.close()
                self.logger.warning(f"Failed to register with node {node_id}")
                
        except Exception as e:
            self.logger.error(f"Error connecting to node {node_id}: {e}")
    
    async def _replicate_node(self) -> None:
        """Replicate node to another host"""
        self.logger.info("Attempting node replication")
        
        with self.lock:
            # Check energy again to avoid race conditions
            if self.state.energy < self.config.replication_threshold:
                self.logger.warning("Insufficient energy for replication")
                return
            
            # Use half energy for replication
            replication_energy = self.state.energy / 2
            self.state.energy -= replication_energy
        
        try:
            # Find suitable hosts for replication
            if self.server_mode:
                # Use network discovery to find hosts
                subnet = self.network_discovery._get_local_subnet()
                live_hosts = self.network_discovery.scan_subnet(subnet)
                
                # Filter out hosts that already have nodes
                candidate_hosts = []
                for host in live_hosts:
                    # Skip localhost
                    if host == "127.0.0.1" or host == "localhost":
                        continue
                    
                    # Skip hosts with known nodes
                    known_host = False
                    for node_info in self.discovered_nodes.values():
                        if node_info.get("ip_address") == host:
                            known_host = True
                            break
                    
                    if not known_host:
                        candidate_hosts.append(host)
                
                # Try to replicate to a random host
                if candidate_hosts:
                    target_host = random.choice(candidate_hosts)
                    
                    # Deploy node to target host
                    success = await self.ssh_deployer.deploy_node(target_host)
                    
                    if success:
                        self.logger.info(f"Successfully replicated to {target_host}")
                        self.state.successful_replications += 1
                    else:
                        self.logger.warning(f"Failed to replicate to {target_host}")
                        
                        # Restore some energy
                        with self.lock:
                            self.state.energy += replication_energy * 0.7
                
                else:
                    self.logger.warning("No suitable hosts found for replication")
                    
                    # Restore some energy
                    with self.lock:
                        self.state.energy += replication_energy * 0.7
            
            else:
                # Client mode - ask server to replicate
                if self.client_connection and self.connected:
                    # Send replication request
                    replication_msg = NetworkMessage(
                        message_type="replication_request",
                        sender_id=self.node_id,
                        content={
                            "energy": replication_energy,
                            "timestamp": time.time()
                        }
                    )
                    
                    await self.client_connection.send(json.dumps(replication_msg.to_dict()))
                    
                    # Count as successful (server will handle actual replication)
                    self.state.successful_replications += 1
                    
                else:
                    self.logger.warning("Not connected to server for replication")
                    
                    # Restore some energy
                    with self.lock:
                        self.state.energy += replication_energy * 0.7
        
        except Exception as e:
            self.logger.error(f"Error in replication: {e}")
            
            # Restore some energy on error
            with self.lock:
                self.state.energy += replication_energy * 0.5
    
    async def _replicate_to_host(self, target_host: str) -> None:
        """Replicate node to a specific host"""
        self.logger.info(f"Replicating to host {target_host}")
        
        try:
            # Deploy node to target host
            success = await self.ssh_deployer.deploy_node(target_host)
            
            if success:
                self.logger.info(f"Successfully replicated to {target_host}")
                self.state.successful_replications += 1
            else:
                self.logger.warning(f"Failed to replicate to {target_host}")
                
        except Exception as e:
            self.logger.error(f"Error replicating to host {target_host}: {e}")
    
    async def _server_maintenance_loop(self) -> None:
        """Server maintenance background task"""
        self.logger.info("Starting server maintenance loop")
        
        while not self.shutdown_event.is_set():
            try:
                # Check for stale nodes
                now = time.time()
                stale_nodes = []
                
                for node_id, attrs in self.network_graph.nodes(data=True):
                    if node_id == self.node_id:
                        continue
                        
                    last_seen = attrs.get("last_seen", 0)
                    if now - last_seen > 60:  # 60 seconds timeout
                        stale_nodes.append(node_id)
                
                # Remove stale nodes
                for node_id in stale_nodes:
                    if node_id in self.network_graph:
                        self.network_graph.remove_node(node_id)
                    
                    if node_id in self.connections:
                        await self.connections[node_id].close()
                        del self.connections[node_id]
                    
                    if node_id in self.state.connected_nodes:
                        self.state.connected_nodes.remove(node_id)
                    
                    if node_id in self.state.entangled_nodes:
                        self.state.entangled_nodes.remove(node_id)
                    
                    self.logger.info(f"Removed stale node {node_id}")
                
                # Update total uptime
                self.state.total_uptime = time.time() - self.state.last_active
                
                await asyncio.sleep(30.0)
                
            except Exception as e:
                self.logger.error(f"Error in server maintenance loop: {e}")
                await asyncio.sleep(30.0)
    
    async def _client_message_loop(self) -> None:
        """Client message processing background task"""
        self.logger.info("Starting client message loop")
        
        while not self.shutdown_event.is_set() and self.connected:
            try:
                # Receive message
                message = await self.client_connection.recv()
                data = json.loads(message)
                
                # Convert to NetworkMessage
                if isinstance(data, dict) and "message_type" in data:
                    msg = NetworkMessage.from_dict(data)
                else:
                    # Legacy message format
                    msg_type = data.get("type", "unknown")
                    msg = NetworkMessage(
                        message_type=msg_type,
                        sender_id=data.get("sender_id", "unknown"),
                        content=data
                    )
                
                # Handle message
                response = await self._handle_message(msg)
                
                # Send response if any
                if response:
                    await self.client_connection.send(json.dumps(response.to_dict()))
                
            except websockets.exceptions.ConnectionClosed:
                self.logger.warning("Connection to server closed")
                self.connected = False
                break
            except Exception as e:
                self.logger.error(f"Error in client message loop: {e}")
                self.connected = False
                break
    
    async def _client_reconnect_loop(self) -> None:
        """Client reconnection background task"""
        self.logger.info("Starting client reconnect loop")
        
        while not self.shutdown_event.is_set():
            # Skip if already connected
            if self.connected:
                await asyncio.sleep(5.0)
                continue
            
            try:
                self.logger.info("Attempting to reconnect to server")
                
                # Connect to server
                self.client_connection = await websockets.connect(self.config.server_url)
                
                # Register with server
                register_msg = NetworkMessage(
                    message_type="register",
                    sender_id=self.node_id,
                    content={
                        "node_name": self.node_name,
                        "energy": self.state.energy,
                        "capabilities": ["compute", "storage", "entanglement"]
                    }
                )
                
                await self.client_connection.send(json.dumps(register_msg.to_dict()))
                
                # Wait for registration acknowledgment
                response = await self.client_connection.recv()
                data = json.loads(response)
                
                if data.get("message_type") == "register_ack":
                    self.connected = True
                    self.logger.info(f"Reconnected to server")
                    
                    # Start message processing
                    self.tasks.add(asyncio.create_task(self._client_message_loop()))
                    
                else:
                    self.logger.error(f"Registration failed: {data}")
                    await self.client_connection.close()
                
            except Exception as e:
                self.logger.error(f"Reconnection error: {e}")
                await asyncio.sleep(10.0)  # Wait longer between reconnect attempts
            
            # Wait before next attempt if not connected
            if not self.connected:
                await asyncio.sleep(10.0)

