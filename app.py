import streamlit as st
import torch
import numpy as np
import rasterio
import cv2
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download
from model import UNet


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="Blessyj/cloud-motion-models",
        filename="unet_model.pt"
    )

    model = UNet(in_channels=6, out_channels=1)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()

def read_geotiff(file, size=(128,128)):
    with rasterio.open(file) as src:
        img = src.read(1).astype(np.float32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = cv2.resize(img, size)

    return img

st.title("Cloud Motion Prediction")
st.write("Predict next cloud frame using INSAT imagery")

uploaded_files = st.file_uploader(
    "Upload 6 GeoTIFF files",
    type=["tif"],
    accept_multiple_files=True
)

if len(uploaded_files) == 6:

    frames = [read_geotiff(f) for f in uploaded_files]

    X = np.array(frames)
    X = np.expand_dims(X, axis=0)

    X_tensor = torch.tensor(X, dtype=torch.float32)

    if st.button("Predict Cloud Motion"):

        with torch.no_grad():
            pred = model(X_tensor)

        pred = pred.squeeze().numpy()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Last Input Frame")
            st.image(frames[-1])

        with col2:
            st.subheader("Predicted Frame")
            st.image(pred)
