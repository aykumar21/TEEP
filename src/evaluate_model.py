import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Define test data path and classes
data_dir = "testing_data"
classes = ['flood', 'non_flood']

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained ResNet-18 model
model = models.resnet18()
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load("flood_detection_model.pth", map_location=torch.device('cpu')))
model.eval()

# Prepare for evaluation
y_true = []
y_pred = []

# Iterate over both classes
for idx, label in enumerate(classes):
    folder = os.path.join(data_dir, label)
    for img_name in os.listdir(folder):
        # Skip hidden/system/non-image files
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(folder, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Skipping file: {img_path} — {e}")
            continue

        image = transform(image).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():
            output = model(image)
            pred = torch.argmax(output, dim=1).item()

        y_true.append(idx)
        y_pred.append(pred)

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=classes))

# Plot and save confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Flood Detection - Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")
