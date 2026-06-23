Cloud Motion Prediction using INSAT-3DR Imagery

 Overview:

Cloud Motion Prediction is a satellite imagery analysis project developed for the ISRO Hackathon. The project focuses on forecasting short-term cloud movement using sequential INSAT-3DR satellite images.
The system processes GeoTIFF satellite imagery, performs image preprocessing, and demonstrates the workflow of cloud motion forecasting using deep learning concepts such as U-Net and Diffusion Models.

 Problem Statement:

Accurate short-term cloud motion forecasting is essential for weather monitoring, disaster management, aviation, and agricultural planning.
This project explores the use of deep learning techniques to predict future cloud formations from a sequence of satellite observations captured by INSAT-3DR.

Features:

* Upload multiple INSAT GeoTIFF cloud images
* Satellite image preprocessing and normalization
* Cloud frame visualization
* Interactive Streamlit web application
* Demonstration of cloud motion prediction workflow
* ISRO Hackathon project prototype

Technologies Used:

* Python
* Streamlit
* PyTorch
* NumPy
* OpenCV
* Rasterio
* Matplotlib
* U-Net Architecture
* Diffusion Models (DDPM)

 Project Structure:

cloud-motion-prediction/

├── app.py

├── model.py

├── utils.py

├── requirements.txt

├── cloud_motion_project.ipynb

└── README.md

How It Works:

1. Upload six GeoTIFF satellite images.
2. Images are normalized and resized.
3. Sequential cloud frames are processed.
4. The system visualizes the latest cloud frame and predicted cloud motion output.
5. Results are displayed through an interactive Streamlit interface.

 Live Demo

Streamlit Application:
https://cloud-motion-prediction-dfjxwfhnzr9hkaksplp42y.streamlit.app/

 GitHub Repository:
https://github.com/Blessyj-0428/cloud-motion-prediction

Future Enhancements:

* Integration of trained U-Net and DDPM models
* Real-time cloud motion forecasting
* Multi-spectral INSAT data support
* Improved prediction accuracy
* Deployment with full inference pipeline

Author

Blessy Margret J
B.Tech Computer Science and Business Systems
Mohmaed Sathak Engineering College,Ramanathapuram,TamilNadu,India

Acknowledgement

 This project was developed as part of an ISRO Hackathon initiative to explore AI-driven approaches for satellite-based weather forecasting and cloud motion prediction.
