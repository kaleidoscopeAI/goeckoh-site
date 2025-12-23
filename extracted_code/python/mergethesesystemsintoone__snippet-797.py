def process_sound(self, audio_data: np.ndarray) -> Dict[str, Any]:
    return {"amplitude": float(np.mean(np.abs(audio_data))), "complexity": float(np.std(audio_data))}

def process_vision(self, image_data: np.ndarray) -> Dict[str, Any]:
    return {"luminance": float(np.mean(image_data)), "contrast": float(np.std(image_data))}

def calculate_dissonance(self, understanding: float, experience: Dict[str, Any]) -> float:
    c = experience.get('complexity', 0.5)
    return min(1.0, abs(understanding - (1.0 - c)) * (1.0 + c))

