import os
import numpy as np
import torch
from tqdm import tqdm

# ===========================================================
# CONFIGURATION
# ===========================================================
DATASET_PATH = "/sr_data2/train_21k/train_lr"
SAVE_DIR = "/sr_data2/dit_sr_f4_multispectral"
os.makedirs(SAVE_DIR, exist_ok=True)

RGB_INDICES = [3, 2, 1] 
HR_SIZE = 96

# ===========================================================
# 1. Setup Data Splits & Stats
# ===========================================================
all_files = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith(".npy")])
train_files = all_files[:10000]
test_files = all_files[10000:11000]

print("Calculating Statistics (from Train set)...")
rgb_pixels = [[] for _ in range(3)]
for fname in tqdm(train_files[:500]): 
    patch = np.load(os.path.join(DATASET_PATH, fname)).astype(np.float32)
    for i, band_idx in enumerate(RGB_INDICES):
        rgb_pixels[i].append(patch[band_idx].flatten())

scales = []
for i in range(3):
    arr = np.concatenate(rgb_pixels[i])
    s = arr.mean() + 2 * arr.std()
    scales.append(s)

# ===========================================================
# 2. Processing Function
# ===========================================================
def extract_and_save(file_list, prefix):
    print(f"Extracting HR Images for {prefix} set...")
    hr_images = []
    
    for fname in tqdm(file_list):
        patch = np.load(os.path.join(DATASET_PATH, fname)).astype(np.float32)
        
        # Pad
        c, h, w = patch.shape
        if h < HR_SIZE or w < HR_SIZE:
            patch = np.pad(patch, ((0,0), (0, max(0, HR_SIZE-h)), (0, max(0, HR_SIZE-w))), mode='reflect')
            
        # Crop
        start_h = (patch.shape[1] - HR_SIZE) // 2
        start_w = (patch.shape[2] - HR_SIZE) // 2
        crop = patch[:, start_h:start_h+HR_SIZE, start_w:start_w+HR_SIZE]
        
        # Extract RGB
        rgb_bands = []
        for i, band_idx in enumerate(RGB_INDICES):
            band = np.clip(crop[band_idx] / scales[i], 0, 1)
            rgb_bands.append(band)
        
        hr_images.append(np.stack(rgb_bands))

    save_path = os.path.join(SAVE_DIR, f"{prefix}_HR_images.npy")
    np.save(save_path, np.array(hr_images))
    print(f"Saved {save_path}")

# ===========================================================
# 3. Run
# ===========================================================
extract_and_save(train_files, "train")
extract_and_save(test_files, "test")


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from diffusers import VQModel
import lpips

# ===========================================================
# CONFIG
# ===========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE = "/sr_data2/dit_sr_f4_multispectral/train_HR_images.npy"
SAVE_DIR = "vqgan_finetuned"
os.makedirs(SAVE_DIR, exist_ok=True)

BATCH_SIZE = 16
LR = 1e-5             # Low learning rate for fine-tuning
EPOCHS = 10           # 5-10 epochs is usually enough
GAN_WEIGHT = 0.1      # Adversarial weight
PERC_WEIGHT = 1.0     # Perceptual weight

# ===========================================================
# 1. Discriminator (PatchGAN)
# ===========================================================
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc=3, ndf=64, n_layers=3):
        super().__init__()
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        return self.main(input)

# ===========================================================
# 2. Dataset
# ===========================================================
class HRImageDataset(Dataset):
    def __init__(self, path):
        self.data = np.load(path) # (N, 3, 96, 96) in [0, 1]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        img = torch.from_numpy(self.data[idx])
        return img * 2.0 - 1.0 # VQGAN expects [-1, 1]

# ===========================================================
# 3. Training Loop
# ===========================================================
def train():
    # Load Data
    ds = HRImageDataset(DATA_FILE)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    # Load Pre-trained VQ-GAN
    print("Loading Pre-trained VQ-GAN...")
    vqgan = VQModel.from_pretrained(
        "CompVis/ldm-super-resolution-4x-openimages",
        subfolder="vqvae"
    ).to(DEVICE)
    vqgan.train()
    
    # Initialize Discriminator
    discriminator = NLayerDiscriminator().to(DEVICE).train()
    
    # Loss (LPIPS)
    lpips_loss = lpips.LPIPS(net='vgg').to(DEVICE).eval()
    
    # Optimizers
    opt_g = torch.optim.Adam(vqgan.parameters(), lr=LR, betas=(0.5, 0.9))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=LR, betas=(0.5, 0.9))
    
    print(f"Starting Fine-Tuning for {EPOCHS} epochs...")
    
    for epoch in range(EPOCHS):
        pbar = tqdm(loader)
        for x in pbar:
            x = x.to(DEVICE)
            
            # ---------------------
            # 1. Train VQ-GAN (Generator)
            # ---------------------
            # FIX: Manual Forward Pass to get commit_loss
            # A. Encode
            h = vqgan.encoder(x)
            h = vqgan.quant_conv(h)
            
            # B. Quantize (Returns: quantized_latents, commit_loss, info)
            quant, commit_loss, _ = vqgan.quantize(h)
            
            # C. Decode
            quant_post = vqgan.post_quant_conv(quant)
            dec = vqgan.decoder(quant_post)
            
            # Reconstruction Loss
            rec_loss = torch.abs(x - dec).mean()
            p_loss = lpips_loss(x, dec).mean()
            
            # Adversarial Loss
            logits_fake = discriminator(dec)
            g_loss = -torch.mean(logits_fake)
            
            # Total Generator Loss
            # We add commit_loss here
            loss_total = rec_loss + PERC_WEIGHT * p_loss + GAN_WEIGHT * g_loss + commit_loss
            
            opt_g.zero_grad()
            loss_total.backward()
            opt_g.step()
            
            # ---------------------
            # 2. Train Discriminator
            # ---------------------
            logits_real = discriminator(x.detach())
            # Use .detach() on dec to avoid backprop to generator
            logits_fake = discriminator(dec.detach())
            
            # Hinge Loss
            loss_real = torch.mean(F.relu(1.0 - logits_real))
            loss_fake = torch.mean(F.relu(1.0 + logits_fake))
            d_loss = 0.5 * (loss_real + loss_fake)
            
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()
            
            pbar.set_description(f"Rec: {rec_loss.item():.3f} | Commit: {commit_loss.item():.3f} | D: {d_loss.item():.3f}")
            
        # Save Checkpoint
        save_path = os.path.join(SAVE_DIR, f"vqgan_epoch_{epoch+1}")
        vqgan.save_pretrained(save_path)
        print(f"Saved fine-tuned model to {save_path}")

if __name__ == "__main__":
    train()


import os
import numpy as np
import torch
from tqdm import tqdm
from diffusers import VQModel

# ===========================================================
# CONFIG
# ===========================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_DIR = "/sr_data2/dit_sr_f4_multispectral"

# Point to your NEW fine-tuned model folder (e.g., epoch 10)
FINE_TUNED_MODEL_PATH = "vqgan_finetuned/vqgan_epoch_10"

# ===========================================================
# Process Function
# ===========================================================
def encode_set(prefix, vae):
    img_path = os.path.join(DATA_DIR, f"{prefix}_HR_images.npy")
    save_path = os.path.join(DATA_DIR, f"{prefix}_HR_latents_2.npy")
    
    if not os.path.exists(img_path):
        print(f"Skipping {prefix}: File not found.")
        return

    print(f"Loading {prefix} images...")
    images = np.load(img_path) 
    
    latents_list = []
    batch_size = 32
    
    print(f"Encoding {prefix}...")
    with torch.no_grad():
        for i in tqdm(range(0, len(images), batch_size)):
            batch_np = images[i:i+batch_size]
            
            # Scale [0, 1] -> [-1, 1] for VQGAN
            batch_t = torch.from_numpy(batch_np).to(DEVICE)
            batch_t = batch_t * 2.0 - 1.0
            
            # Encode
            lat = vae.encode(batch_t).latents
            latents_list.append(lat.cpu().numpy())
            
    all_latents = np.concatenate(latents_list, axis=0)
    np.save(save_path, all_latents)
    print(f"Saved {save_path} shape: {all_latents.shape}")

# ===========================================================
# Main
# ===========================================================
if __name__ == "__main__":
    print(f"Loading Fine-Tuned VQ-GAN from {FINE_TUNED_MODEL_PATH}...")
    # Ensure this path exists!
    if not os.path.exists(FINE_TUNED_MODEL_PATH):
        print("Error: Fine-tuned model path not found. Please run finetune_vqgan.py first.")
        exit()
        
    vae = VQModel.from_pretrained(FINE_TUNED_MODEL_PATH).to(DEVICE).eval()
    
    encode_set("train", vae)
    encode_set("test", vae)
