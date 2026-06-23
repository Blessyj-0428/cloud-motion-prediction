import streamlit as st
import numpy as np
import rasterio
import cv2

st.set_page_config(page_title="Cloud Motion Prediction", layout="wide")

def read_geotiff(file, size=(128, 128)):
    with rasterio.open(file) as src:
        img = src.read(1).astype(np.float32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img = cv2.resize(img, size)

    return img

st.title("🌩️ Cloud Motion Prediction using INSAT-3DR Imagery")
st.write(
    "Demo application developed for the ISRO Hackathon project "
    "'Chase the Cloud: Leveraging Diffusion Models for Cloud Motion Prediction'."
)

uploaded_files = st.file_uploader(
    "Upload 6 GeoTIFF cloud images",
    type=["tif"],
    accept_multiple_files=True
)

if len(uploaded_files) == 6:

    frames = [read_geotiff(f) for f in uploaded_files]

    st.success("6 GeoTIFF files uploaded successfully!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Latest Input Cloud Frame")
        st.image(frames[-1], use_container_width=True)

    with col2:
        st.subheader("Predicted Cloud Motion (Demo)")
        predicted = cv2.GaussianBlur(frames[-1], (9, 9), 0)
        st.image(predicted, use_container_width=True)

    st.info(
        "Portfolio Demo Version: Displays cloud-frame processing workflow "
        "for the ISRO Hackathon project."
    )

else:
    st.warning("Please upload exactly 6 GeoTIFF files.")
