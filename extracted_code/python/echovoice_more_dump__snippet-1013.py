"""Network discovery and node detection"""

def __init__(self, config: QSINConfig):
    self.config = config
    self.discovered_nodes = {}  # ip -> node_info
    self.successful_scans = 0

def scan_subnet(self, subnet: str) -> List[str]:
    """Scan subnet for potential nodes"""
    if not subnet:
        # Try to determine local subnet
        subnet = self._get_local_subnet()

    logger.info(f"Scanning subnet: {subnet}")

    try:
        # Simple ping scan implementation
        if "/" in subnet:  # CIDR notation
            base, prefix = subnet.split("/")
            octets = base.split(".")
            base_ip = ".".join(octets[:3])

            live_hosts = []
            for i in range(1, 255):
                ip = f"{base_ip}.{i}"
                if self._ping_host(ip):
                    live_hosts.append(ip)

            self.successful_scans += 1
            return live_hosts
        else:
            return [subnet] if self._ping_host(subnet) else []
    except Exception as e:
        logger.error(f"Subnet scan error: {e}")
        return []

def _get_local_subnet(self) -> str:
    """Get local subnet for scanning"""
    try:
        # Get local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Doesn't need to be reachable, just to determine interface
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()

        # Extract subnet
        octets = local_ip.split(".")
        subnet = f"{octets[0]}.{octets[1]}.{octets[2]}.0/24"
        return subnet
    except:
        # Fallback to common private subnet
        return "192.168.1.0/24"

def _ping_host(self, ip: str) -> bool:
    """Check if host is reachable via ping"""
    try:
        # Use subprocess to ping with timeout
        param = "-n" if os.name == "nt" else "-c"
        cmd = ["ping", param, "1", "-W", "1", ip]
        result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return result.returncode == 0
    except:
        return False

def check_for_node(self, ip: str, port: int = DEFAULT_PORT) -> Optional[Dict[str, Any]]:
    """Check if IP has a QSIN node running"""
    try:
        # Try to connect to WebSocket
        ws_url = f"ws://{ip}:{port}/qsin"

        # Create connection with short timeout
        loop = asyncio.get_event_loop()
        try:
            # Connect with 2 second timeout
            ws = await asyncio.wait_for(
                websockets.connect(ws_url),
                timeout=2.0
            )

            # Send discovery ping
            discovery_msg = {
                "type": "discovery_ping",
                "source_id": self.config.node_id,
                "source_name": self.config.node_name,
                "timestamp": time.time()
            }

            await ws.send(json.dumps(discovery_msg))

            # Wait for response
            response = await asyncio.wait_for(
                ws.recv(),
                timeout=2.0
            )

            # Parse response
            data = json.loads(response)
            if data.get("type") == "discovery_pong":
                node_info = {
                    "node_id": data.get("node_id", "unknown"),
                    "node_name": data.get("node_name", "unknown"),
                    "ip_address": ip,
                    "port": port,
                    "last_seen": time.time(),
                    "capabilities": data.get("capabilities", [])
                }

                # Store discovered node
                self.discovered_nodes[ip] = node_info

                # Close connection
                await ws.close()

                return node_info
        except asyncio.TimeoutError:
            # Connection timeout
            return None
        except Exception as e:
            # Other connection error
            return None

    except Exception as e:
        logger.debug(f"Error checking for node at {ip}:{port}: {e}")
        return None

async def discover_nodes(self, subnet: Optional[str] = None) -> List[Dict[str, Any]]:
    """Discover QSIN nodes on the network"""
    # Scan subnet for live hosts
    live_hosts = self.scan_subnet(subnet or "")

    # Check each host for QSIN node
    discovered = []

    # Use gather with limit to avoid too many concurrent connections
    tasks = []
    for ip in live_hosts:
        task = asyncio.create_task(self.check_for_node(ip))
        tasks.append(task)

        # Limit concurrent tasks
        if len(tasks) >= 10:
            # Wait for some tasks to complete
            done, tasks = await asyncio.wait(
                tasks, 
                return_when=asyncio.FIRST_COMPLETED
            )

            # Process completed tasks
            for task in done:
                result = task.result()
                if result:
                    discovered.append(result)

    # Wait for remaining tasks
    if tasks:
        done, _ = await asyncio.wait(tasks)
        for task in done:
            result = task.result()
            if result:
                discovered.append(result)

    logger.info(f"Discovered {len(discovered)} QSIN nodes on network")
    return discovered

