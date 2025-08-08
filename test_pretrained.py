import os
import rasterio
from rasterio.windows import Window
import torch
import torch.nn as nn
import torchvision.transforms as T
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F
from functools import partial
from torchvision import models

# DinkNet34 model definition (copied from your networks/dinknet.py)
nonlinearity = partial(F.relu, inplace=True)

class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()
                    
    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class DinkNet34(nn.Module):
    def __init__(self, num_classes=1, num_channels=3):
        super(DinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        
        self.dblock = Dblock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        
        # Center
        e4 = self.dblock(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)
        
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

def predict_large_image(model_path, image_path, output_path, patch_size=256, batch_size=8):
    """
    Applies a trained DinkNet34 model to a large satellite image and saves the output.
    """
    # --- 1. Load the Model ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create the DinkNet34 model
    model = DinkNet34().to(device)
    
    # Wrap in DataParallel (matching the training setup)
    model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    
    # Load the saved state dictionary
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

    # Set model to evaluation mode
    model.eval()

    # --- 2. Define Image Transformations ---
    # Note: DinkNet34 expects BGR format and specific normalization
    # Based on the inference code: (input / 255.0 * 3.2 - 1.6)
    def preprocess_patch(patch_rgb):
        # Convert RGB to BGR (matching the training preprocessing)
        patch_bgr = patch_rgb[:, :, ::-1].copy()  # RGB to BGR with copy to fix stride issue
        
        # Convert to tensor and normalize like in the original inference
        tensor = torch.from_numpy(patch_bgr).float().permute(2, 0, 1) / 255.0 * 3.2 - 1.6
        return tensor.unsqueeze(0)

    # --- 3. Process the Large Image in Patches ---
    with rasterio.open(image_path) as src:
        print(f"Processing image of size: {src.width}x{src.height}")
        
        # Get metadata from the source image
        meta = src.meta.copy()
        # Update metadata for a single-band (grayscale) output
        meta.update(count=1, dtype='uint8', compress='lzw', nodata=0)

        with rasterio.open(output_path, 'w', **meta) as dst:
            # Iterate over the image in patches (windows)
            total_patches = ((src.height - 1) // patch_size + 1) * ((src.width - 1) // patch_size + 1)
            print(f"Total patches to process: {total_patches}")
            
            patch_count = 0
            for j in tqdm(range(0, src.height, patch_size), desc="Processing Rows"):
                for i in tqdm(range(0, src.width, patch_size), desc="Processing Columns", leave=False):
                    patch_count += 1
                    
                    window = Window(i, j, min(patch_size, src.width - i), min(patch_size, src.height - j))

                    # Read a patch from the source image
                    patch = src.read(window=window)

                    # Ensure patch has 3 bands
                    patch = patch[:3, :, :]
                    
                    # Skip empty patches
                    if patch.shape[1] == 0 or patch.shape[2] == 0:
                        continue
                    
                    # Convert from (bands, height, width) to (height, width, bands)
                    patch_rgb = np.moveaxis(patch, 0, -1)
                    
                    # Resize patch to standard size if needed (DinkNet typically expects fixed size)
                    if patch_rgb.shape[:2] != (patch_size, patch_size):
                        import cv2
                        patch_rgb = cv2.resize(patch_rgb, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
                    
                    # Ensure uint8 format
                    if patch_rgb.dtype != np.uint8:
                        # Normalize to 0-255 range if needed
                        patch_rgb = ((patch_rgb - patch_rgb.min()) / (patch_rgb.max() - patch_rgb.min()) * 255).astype(np.uint8)
                    
                    # Preprocess the patch
                    input_tensor = preprocess_patch(patch_rgb).to(device)

                    # --- 4. Make Predictions ---
                    with torch.no_grad():
                        # DinkNet34 outputs sigmoid directly
                        output = model(input_tensor)
                        # Apply threshold to get binary prediction
                        preds = output > 0.5

                    # Convert prediction tensor to numpy array
                    mask = preds.cpu().numpy().squeeze().astype(np.uint8) * 255
                    
                    # Resize mask back to original patch size if it was resized
                    if mask.shape != (window.height, window.width):
                        import cv2
                        mask = cv2.resize(mask, (window.width, window.height), interpolation=cv2.INTER_NEAREST)
                    
                    # Create output array with correct dimensions
                    output_mask = np.zeros((window.height, window.width), dtype=np.uint8)
                    output_mask[:mask.shape[0], :mask.shape[1]] = mask

                    # --- 5. Write the Prediction to the Output Raster ---
                    dst.write(output_mask[np.newaxis, :, :], window=window)
                    
                    if patch_count % 100 == 0:
                        print(f"Processed {patch_count}/{total_patches} patches")

    print(f"Prediction saved to {output_path}")

# --- Set Your Paths ---
model_path = r"D:\Road-extraction-model\log01_dink34.th"
large_image_path = r"D:\Road-extraction-model\img_50cm.tif"
output_prediction_path = r"D:\Road-extraction-model\output_prediction_pretrained.tif"

# --- Run the Prediction ---
if __name__ == "__main__":
    predict_large_image(model_path, large_image_path, output_prediction_path, patch_size=256)