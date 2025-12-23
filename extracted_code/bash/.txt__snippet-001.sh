# native install
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
# pull DeepSeek model in Ollama (host)
ollama pull deepseek-r1:8b
# record child voice
python record_voice_sample.py -d 6.0
# run server
python echo_server.py
Dashboard: http://127.0.0.1:5000
Mobile API: POST audio/wav → http://SERVER_IP:5000/api/utterance
Docker
Printed using ChatGPT to PDF, powered by PDFCrowd HTML to PDF API.
53/95bash
docker compose up -d
Then:
Dashboard: http://SERVER_IP:5000
Ollama endpoint (inside container): http://ollama:11434
EchoMobile (iOS)
Swift files:
EchoMobileApp.swift
ServerConfig.swift
EchoClient.swift
RootView.swift
ContentView.swift
ServerSettingsView.swift
Info.plist keys:
xml
<key>NSMicrophoneUsageDescription</key>
<string>This app uses the microphone so Echo can listen and respond in my voice.</string>
<key>NSAppTransportSecurity</key>
<dict>
<key>NSAllowsArbitraryLoads</key>
<true/>
</dict>
Update ServerConfig host/port or via in-app settings.
bash
