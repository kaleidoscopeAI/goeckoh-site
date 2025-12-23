def __init__(self, config: QSINConfig):
    self.config = config
    self.energy = config.initial_energy
    self.server_connection = None
    self.is_running = False
    self.connected = False

async def connect_to_server(self):
    try:
        self.server_connection = await websockets.connect(self.config.server_url)

        # Register with server
        register_msg = {
            "type": "register",
            "node_id": self.config.node_id,
            "node_name": self.config.node_name,
            "energy": self.energy,
            "timestamp": time.time()
        }
        await self.server_connection.send(json.dumps(register_msg))

        # Wait for response
        response = await self.server_connection.recv()
        data = json.loads(response)

        if data.get("type") == "register_ack":
            self.connected = True
            logger.info(f"Connected to QSIN server as {self.config.node_name}")
            return True
        else:
            logger.error(f"Registration failed: {data}")
            return False
    except Exception as e:
        logger.error(f"Connection error: {e}")
        self.connected = False
        return False

async def process_energy(self):
    """Simulate energy processing and growth"""
    while self.is_running:
        # Increase energy over time
        self.energy += random.uniform(1.0, 3.0)

        # Report energy to server
        if self.connected:
            try:
                update_msg = {
                    "type": "energy_update",
                    "node_id": self.config.node_id,
                    "energy": self.energy,
                    "timestamp": time.time()
                }
                await self.server_connection.send(json.dumps(update_msg))
            except:
                self.connected = False

        # Sleep for a bit
        await asyncio.sleep(5)

async def process_messages(self):
    """Process messages from the server"""
    while self.is_running and self.connected:
        try:
            message = await self.server_connection.recv()
            data = json.loads(message)

            # Handle message based on type
            msg_type = data.get("type", "")

            if msg_type == "ping":
                # Respond to ping
                pong_msg = {
                    "type": "pong",
                    "node_id": self.config.node_id,
                    "timestamp": time.time()
                }
                await self.server_connection.send(json.dumps(pong_msg))

            elif msg_type == "task_request":
                # Simulate task processing
                task_id = data.get("task_id", "")

                # Process task (simulate computation)
                await asyncio.sleep(random.uniform(0.5, 2.0))

                # Send result
                result_msg = {
                    "type": "task_result",
                    "node_id": self.config.node_id,
                    "task_id": task_id,
                    "result": random.random(),
                    "timestamp": time.time()
                }
                await self.server_connection.send(json.dumps(result_msg))

            elif msg_type == "discovery_ping":
                # Respond to discovery ping
                pong_msg = {
                    "type": "discovery_pong",
                    "node_id": self.config.node_id,
                    "node_name": self.config.node_name,
                    "timestamp": time.time(),
                    "capabilities": ["compute", "storage"]
                }
                await self.server_connection.send(json.dumps(pong_msg))

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            self.connected = False
            break

async def reconnect_loop(self):
    """Try to reconnect if connection is lost"""
    while self.is_running:
        if not self.connected:
            logger.info("Attempting to reconnect...")
            await self.connect_to_server()

        # Wait before checking again
        await asyncio.sleep(5)

async def run(self):
    """Run the node"""
    self.is_running = True

    # Connect to server
    connected = await self.connect_to_server()
    if not connected:
        logger.error("Failed to connect to server")
        return

    # Start background tasks
    tasks = [
        asyncio.create_task(self.process_energy()),
        asyncio.create_task(self.process_messages()),
        asyncio.create_task(self.reconnect_loop())
    ]

    # Wait for tasks to complete (or error)
    await asyncio.gather(*tasks, return_exceptions=True)

def stop(self):
    """Stop the node"""
    self.is_running = False

