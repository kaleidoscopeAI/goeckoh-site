import ollama

class OllamaNode(Node):
    def __init__(self, id, dim=3):
        super().__init__(id, dim)
        self.is_ollama_node = True
        
    def process_text(self, text):
        try:
            response = ollama.chat(model='llama2', messages=[
                {
                    'role': 'user',
                    'content': f"Summarize this text and extract key entities:\n\n{text}",
                },
            ])
            return response['message']['content']
        except Exception as e:
            print(f"Ollama processing error: {e}")
            return ""

