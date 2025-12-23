    28      private initializeWebSocket() {
    29 -        // Connect to backend server on port 5000
    29 +        // Optional WebSocket; keep existing but allow override
    30          const protocol = window.location.protocol === 'https:' ? 'wss:'
        : 'ws:';

