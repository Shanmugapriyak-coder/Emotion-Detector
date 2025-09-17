
import streamlit as st 
import torch
# from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
from torch import nn,optim
import joblib
from torchvision import transforms

emotion_list=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise']

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # if the model expects grayscale
    transforms.Resize((48, 48)),                  # resize to model's input size
    transforms.ToTensor(),                        # convert to tensor [C, H, W], values in [0,1]
    # transforms.Normalize((0.5,), (0.5,))         # optional: normalize for grayscale
])

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),   # [32, 48, 48]
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),                           # [32, 24, 24]

            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # [64, 24, 24]
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),                           # [64, 12, 12]

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # [128, 12, 12]
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2)                            # [128, 6, 6]
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 7)  # 7 output classes for emotions
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.fc(x)
        return x

# Instantiate the model
model = CNN()
model.load_state_dict(torch.load(r'C:\Users\MY Laptop\Desktop\guvi_class\emotion detection app\emotion_cnn_model.pth'))

# Set Streamlit page config
st.set_page_config(page_title="Facial Emotion Detector", layout="centered")

# Custom CSS styles
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
        }
        h1 {
            color: #2c3e50;
            text-align: center;
        }
        .footer {
            font-size: 13px;
            color: #888;
            text-align: center;
            margin-top: 40px;
        }
    </style>
""", unsafe_allow_html=True)

# Title and Instructions
st.markdown("<div class='main'>", unsafe_allow_html=True)
st.title("🖤 Facial Emotion Detector")
st.markdown("Upload a image .")

# Upload image
uploaded_file = st.file_uploader("📷 Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale
    st.image(image, caption="🖼 Uploaded Image", use_container_width=True)

    # Resize and preprocess image
    # image = Image.open("path_to_your_image.jpg")  # replace with your uploaded image path
    img_tensor = transform(image)                 # shape: [1, 48, 48]
    img_tensor = img_tensor.unsqueeze(0)          # add batch dimension: [1, 1, 48, 48]

   

    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
         prediction = model(img_tensor)

    predicted_class = np.argmax(prediction)

    st.success(f"✅ **Detected Emotion:** {emotion_list[predicted_class]}")
    # st.bar_chart(prediction[0])
