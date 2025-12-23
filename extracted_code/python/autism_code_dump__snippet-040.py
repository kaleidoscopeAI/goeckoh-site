class APIServerHandler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: Dict[str, Any]):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(payload).encode("utf-8"))

    def do_GET(self):
        global RUNNING
        if self.path == "/health":
            self._send_json(200, {"status": "ok", "running": RUNNING})
        elif self.path == "/kill":
            RUNNING = False
            self._send_json(200, {"status": "stopping"})
        elif self.path == "/wipe":
            for p in (ATTEMPTS_CSV, GUIDANCE_CSV):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass
            for wav in VOICES_DIR.glob("*.wav"):
                try:
                    wav.unlink()
                except Exception:
                    pass
            self._send_json(200, {"status": "wiped"})
        else:
            self._send_json(404, {"error": "not found"})

    def log_message(self, format, *args):
        return  # silence default HTTP logging


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_api_server(host: str = "127.0.0.1", port: int = 8081):
    server = ThreadedHTTPServer((host, port), APIServerHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    print(f"[API] http://{host}:{port} (kill: /kill, wipe: /wipe)")
    return server


