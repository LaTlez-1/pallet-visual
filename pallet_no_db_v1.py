import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Load your model
model = torch.jit.load("v3torchscript.pt")
model.eval()

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Map prediction keys to labels
prediction_labels = {0: 'A', 1: 'B', 2: 'C'}

st.title("แอปพลิเคชันคัดแยกไม้พาเลท")

def classify_image(image):
    image = transform(image).unsqueeze(0)  # Transform and add batch dimension
    with torch.no_grad():
        output = model(image)
    _, predicted = torch.max(output, 1)
    return prediction_labels[predicted.item()]

# Layout for 2x2 image uploaders with specific names
frame_names = ['ด้านหน้า', 'ด้านหลัง', 'ด้านซ้าย', 'ด้านขวา']
cols = st.columns(2)

for i in range(2):
    for j in range(2):
        frame_name = frame_names[i*2 + j]
        with cols[j]:
            st.header(frame_name)
            uploaded_file = st.file_uploader(f"Upload {frame_name} image", type=["jpg", "jpeg", "png"], key=frame_name)

            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert("RGB")
                st.image(image, caption=f"{frame_name} image", use_column_width=True)
                
                # Perform classification
                prediction = classify_image(image)
                st.write(f"ประเภทไม้พาเลทสำหรับ {frame_name}: {prediction}")
