import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
from torchvision import models
import joblib
import sys
import os

if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "card_cnn_model.pth")
encoder_path = os.path.join(BASE_DIR, "label_encoder.joblib")

num_classes = 53
model = models.resnet18(pretrained=False)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

label_encoder = joblib.load(encoder_path)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

root = tk.Tk()
root.title("🎴 Card Identifier")
root.geometry("600x500")
root.configure(bg="#f5f5f5")

header = tk.Label(root, text="🎴 Card Identifier", font=("Arial", 18, "bold"), bg="#f5f5f5", fg="#333")
header.pack(pady=20)

img_panel = tk.Label(root, bg="#f5f5f5")
img_panel.pack()

result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f5f5f5", fg="blue")
result_label.pack(pady=20)

def upload_and_classify():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    if not file_path:
        return
    try:
        img = Image.open(file_path).convert("RGB")
        display_img = img.resize((250, 250))
        img_tk = ImageTk.PhotoImage(display_img)
        img_panel.config(image=img_tk)
        img_panel.image = img_tk

        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = label_encoder.inverse_transform([pred])[0]

        result_label.config(text=f"Predicted: {label}")

    except Exception as e:
        messagebox.showerror("Error", f"Failed to classify image.\n\n{e}")

btn = tk.Button(root, text="📁 Upload Card Image", command=upload_and_classify, font=("Arial", 12), bg="#0078D7", fg="white")
btn.pack(pady=10)

if __name__ == "__main__":
    root.mainloop()
