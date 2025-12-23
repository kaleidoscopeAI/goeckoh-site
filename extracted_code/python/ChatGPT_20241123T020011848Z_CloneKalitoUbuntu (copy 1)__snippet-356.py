def __init__(self):
    self.resources = {
        "pretrained_model": "YOLOv4 weights",
        "text_api": "NLP API",
        "image_annotator": "OpenCV-based annotator",
    }

def get_resource(self, resource_name):
    return self.resources.get(resource_name, "Resource not available.")

