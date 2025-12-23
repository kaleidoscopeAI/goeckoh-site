GraphNeuralNetwork (GNN): If PyTorch and PyTorch Geometric are available, this model is used to learn relationships and make predictions based on the graph structure of interacting nodes. It takes chemical features and positions as input.

SelfCorrectionModel (Variational Autoencoder - VAE): If PyTorch is available, this model attempts to learn a latent representation of the nodes' states and chemical properties, enabling self-correction or anomaly detection within the system.

SystemVoice: Utilizes pyttsx3 for text-to-speech output, providing audible alerts and summaries of system events (e.g., high node stress, new insights, simulation status). It processes events via a separate thread to avoid blocking the main simulation.

VisualInputProcessor: Integrates with OpenCV (cv2) to potentially process video input (from a webcam or file). In the current implementation, it can detect simple "visual anomalies" to generate insights.

WebSocketClient: Allows the system to receive real-time data or commands from an external WebSocket server.

QueryParser: A basic Natural Language Processing (NLP) component that can parse simple user queries like "Find compounds similar to aspirin but safer for the stomach," identifying target SMILES and molecular property constraints.

