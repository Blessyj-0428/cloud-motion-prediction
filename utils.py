
import torch
import numpy as np
import rasterio
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_absolute_error


def load_frames(files):
    stack = []
    for file in files:
        with rasterio.open(file) as src:
            frame = src.read(1).astype(np.float32) / 255.0  # Normalize
            stack.append(frame)
    input_tensor = np.stack(stack)  # Shape: (6, H, W)
    return torch.tensor(input_tensor).unsqueeze(0)  # (1, 6, H, W)


def load_unet(path="models/unet_model.pt", device="cpu"):
    from app import UNet  # assumes UNet is defined in app.py
    model = UNet(in_channels=6, out_channels=1)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def load_ddpm(path="models/ddpm_model.pt", device="cpu"):
    from app import UNet, DDPM  # assumes DDPM is defined in app.py
    unet = UNet(in_channels=6, out_channels=1)
    ddpm = DDPM(denoise_model=unet)
    ddpm.load_state_dict(torch.load(path, map_location=device))
    ddpm.eval()
    return ddpm


def predict_unet(input_tensor, model):
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).squeeze(0).cpu().numpy()
    return (output * 255).astype(np.uint8)

def predict_ddpm(input_tensor, model):
    with torch.no_grad():
        generated = model.sample(shape=(1, 1, input_tensor.shape[2], input_tensor.shape[3]), device="cpu")
    return (generated.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)


def compute_metrics(pred, truth):
    pred = pred.astype(np.float32) / 255.0
    truth = truth.astype(np.float32) / 255.0
    return (
        ssim(truth, pred, data_range=1.0),
        psnr(truth, pred, data_range=1.0),
        mean_absolute_error(truth.flatten(), pred.flatten())
    )
