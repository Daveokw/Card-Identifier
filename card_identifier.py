import os
import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import joblib
import sys

# Setup paths
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "card_cnn_model.pth")
encoder_path = os.path.join(BASE_DIR, "label_encoder.joblib")

# Load model
num_classes = 53
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

# Load label encoder
label_encoder = joblib.load(encoder_path)

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Streamlit UI
st.set_page_config(page_title="🎴 Card Identifier", layout="centered")
st.title("🎴 Card Identifier")
st.write("Upload a card image to identify its type.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=False, width=250)

        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = label_encoder.inverse_transform([pred])[0]

        st.success(f"🧠 Predicted: **{label}**")
    except Exception as e:
        st.error(f"❌ Error during classification: {e}")
