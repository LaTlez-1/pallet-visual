import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import io

import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase


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

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame_name = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        image = Image.fromarray(img)
        image = image.convert("RGB")
        
        prediction = classify_image(image)
        st.write(f"ประเภทไม้พาเลทสำหรับ {self.frame_name}: {prediction}")
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Layout for 2x2 frames with specific names
frame_names = ['ด้านหน้า', 'ด้านหลัง', 'ด้านซ้าย', 'ด้านขวา']
cols = st.columns(2)

for i in range(2):
    for j in range(2):
        frame_name = frame_names[i*2 + j]
        with cols[j]:
            st.header(frame_name)
            webrtc_ctx = webrtc_streamer(
                key=f"camera_input_{frame_name}",
                video_transformer_factory=VideoTransformer,
                media_stream_constraints={"video": True, "audio": False},
            )
            if webrtc_ctx.video_transformer:
                webrtc_ctx.video_transformer.frame_name = frame_name