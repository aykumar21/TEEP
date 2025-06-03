import torch
from torchvision import models, transforms
from PIL import Image
import sys

# Load model architecture
def get_model():
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    return model

# Load the model
model = get_model()
model.load_state_dict(torch.load("flood_detection_model.pth", map_location=torch.device('cpu')))
model.eval()

# Image input from command line
if len(sys.argv) < 2:
    print("Usage: python3 predict_flood.py <image_path>")
    exit()

image_path = sys.argv[1]

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

img = Image.open(image_path)
input_tensor = transform(img).unsqueeze(0)

# Prediction
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)

result = "Flooded" if predicted.item() == 1 else "Not Flooded"
print(f"Prediction: {result}")
