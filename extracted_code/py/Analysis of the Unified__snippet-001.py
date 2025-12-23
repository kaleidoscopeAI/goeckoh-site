# server.py
from flask import Flask, send_from_directory, jsonify
from flask_socketio import SocketIO, emit
import threading, time, numpy as np, requests, json
from collections import deque

app = Flask(__name__, static_folder='client')
socketio = SocketIO(app, cors_allowed_origins="*")

NODE_COUNT = 20000  # lower default for server demo
positions = np.zeros((NODE_COUNT, 3), dtype=np.float32)
velocities = (np.random.rand(NODE_COUNT, 3).astype(np.float32) - 0.5) * 2.0
for i in range(NODE_COUNT):
    phi = np.arccos(1 - 2 * (i + 0.5) / NODE_COUNT)
    theta = np.pi * 2 * (i + 0.5) / (1 + 5**0.5)
    positions[i,0] = 400 * np.sin(phi) * np.cos(theta)
    positions[i,1] = 400 * np.sin(phi) * np.sin(theta)
    positions[i,2] = 400 * np.cos(phi)

# simple ring buffer for snapshots for replay
snapshots = deque(maxlen=1000)

OLLAMA_PROXY = 'http://localhost:5174/api/generate'

def physics_loop():
    global positions, velocities
    step = 0
    while True:
        t0 = time.time()
        # simple physics: attraction to origin modulated by "emotional field"
        dist = np.linalg.norm(positions, axis=1) + 1e-6
        forces = -positions / dist[:,None] * (1.0 / (dist[:,None] + 1.0))[:,None] * 0.02
        velocities += forces
        velocities *= 0.98
        positions += velocities * 0.016
        # boundary
        r = np.linalg.norm(positions, axis=1)
        mask = r > 400
        positions[mask] *= 0.95
        velocities[mask] *= 0.95
        # snapshot for replay
        snapshots.append(positions.copy())
        # broadcast to clients (pack as bytes to reduce JSON overhead)
        try:
            socketio.emit('positions', {'positions': positions.tobytes()}, broadcast=True, namespace='/kaleido')
        except Exception as e:
            print('emit error', e)
        step += 1
        dt = time.time() - t0
        sleep_time = max(0.0, 0.016 - dt)
        time.sleep(sleep_time)

@app.route('/')
def index():
    return send_from_directory('client', 'index.html')

@app.route('/health')
def health():
    return jsonify({'status':'ok'})

@socketio.on('connect', namespace='/kaleido')
def handle_connect():
    print('client connected')
    emit('hello', {'msg':'welcome'})

def start_physics_thread():
    th = threading.Thread(target=physics_loop, daemon=True)
    th.start()

if __name__ == '__main__':
    start_physics_thread()
    socketio.run(app, host='0.0.0.0', port=5000)
