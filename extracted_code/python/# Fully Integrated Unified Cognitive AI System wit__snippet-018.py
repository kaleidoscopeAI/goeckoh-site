import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import numpy as np

# Load a pre-trained model for embedding
model = models.resnet50(pretrained=True).eval()
preprocess = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def embed_image(image_path):
    img = Image.open(image_path)
    img_t = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        features = model.avgpool(model.layer4(model.layer3(model.layer2(model.layer1(model.relu(model.bn1(model.conv1(img_t))))))))
        features = features.squeeze()
    return features.cpu().numpy()
