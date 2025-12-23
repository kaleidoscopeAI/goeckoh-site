class ResourceLibrary:
    def __init__(self):
        self.resources = {
            "object_detection": "pretrained_yolo_model",
            "text_generation": "pretrained_gpt_model",
            "image_annotation": "annotation_api",
        }

    def get_resource(self, resource_name):
        return self.resources.get(resource_name, "Resource not available.")

