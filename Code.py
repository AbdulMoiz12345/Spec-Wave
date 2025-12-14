import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===========================================================
# 1. Configuration & Constants
# ===========================================================
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

BASE_CHANNELS = 192          
TRANSFORMER_BLOCKS = [6, 6, 6, 6] 
WINDOW_SIZE = 4              
LATENT_CHANNELS = 3          
NUM_SPECTRAL_BANDS = 12      
TIME_DIM = 128               

# ===========================================================
# 2. Wavelet Helpers
# ===========================================================
def haar_dwt(x):
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        x = F.pad(x, (0, W % 2, 0, H % 2), mode='reflect')
    
    x00 = x[:, :, 0::2, 0::2]
    x01 = x[:, :, 0::2, 1::2]
    x10 = x[:, :, 1::2, 0::2]
    x11 = x[:, :, 1::2, 1::2]
    
    x_LL = (x00 + x01 + x10 + x11) / 2
    x_HL = (x00 - x10 + x01 - x11) / 2
    x_LH = (x00 + x10 - x01 - x11) / 2
    x_HH = (x00 - x10 - x01 + x11) / 2
    
    return torch.cat([x_LL, x_HL, x_LH, x_HH], dim=1)

def haar_idwt(x):
    B, C4, H2, W2 = x.shape
    C = C4 // 4
    x_LL = x[:, 0:C, :, :]
    x_HL = x[:, C:2*C, :, :]
    x_LH = x[:, 2*C:3*C, :, :]
    x_HH = x[:, 3*C:4*C, :, :]
    
    x00 = (x_LL + x_HL + x_LH + x_HH) / 2
    x01 = (x_LL + x_HL - x_LH - x_HH) / 2
    x10 = (x_LL - x_HL + x_LH - x_HH) / 2
    x11 = (x_LL - x_HL - x_LH + x_HH) / 2
    
    out = torch.zeros(B, C, H2*2, W2*2, device=x.device)
    out[:, :, 0::2, 0::2] = x00
    out[:, :, 0::2, 1::2] = x01
    out[:, :, 1::2, 0::2] = x10
    out[:, :, 1::2, 1::2] = x11
    return out

# ===========================================================
# 3. Core Modules: Time & AdaWM
# ===========================================================
def timestep_embedding(timesteps, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2: emb = F.pad(emb, (0, 1))
    return emb

class AdaWM(nn.Module):
    """ Fixed Adaptive Wavelet Modulation """
    def __init__(self, channels, time_dim=TIME_DIM):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, channels * 4) 
        )

    def forward(self, x, time_emb):
        B, C, H, W = x.shape
        x_dwt = haar_dwt(x)
        scale = self.mlp(time_emb).view(B, -1, 1, 1)
        x_mod = x_dwt * scale
        x_out = haar_idwt(x_mod)
        
        # Crop back if needed (Handle odd dimensions)
        if x_out.shape[2] != H or x_out.shape[3] != W:
            x_out = x_out[:, :, :H, :W]
        return x_out

# ===========================================================
# 4. Advanced Transformer Components (GDFN, SE, Attn)
# ===========================================================
class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size, shift_size=0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, C, H, W = x.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w: x = F.pad(x, (0, pad_w, 0, pad_h))
        _, _, Hp, Wp = x.shape

        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(2, 3))

        x_windows = x.view(B, C, Hp // ws, ws, Wp // ws, ws).permute(0, 2, 4, 3, 5, 1).contiguous().view(-1, ws*ws, C)

        qkv = self.qkv(x_windows)
        qkv = qkv.reshape(-1, ws*ws, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(-1, ws*ws, C)
        out = self.proj(out)

        out = out.view(B, Hp // ws, Wp // ws, ws, ws, C).permute(0, 5, 1, 3, 2, 4).contiguous().view(B, C, Hp, Wp)

        if self.shift_size > 0:
            out = torch.roll(out, shifts=(self.shift_size, self.shift_size), dims=(2, 3))

        if pad_h or pad_w: out = out[:, :, :H, :W]
        return out

class ChannelAttention(nn.Module):
    """ Squeeze-and-Excitation (SE) Block """
    def __init__(self, dim, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, dim, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class GatedDconvFeedForward(nn.Module):
    """ GDFN: Gated Depthwise-Conv Feed-Forward Network """
    def __init__(self, dim, expansion_factor=2.66):
        super().__init__()
        hidden_dim = int(dim * expansion_factor)
        
        self.project_in = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_dim * 2, hidden_dim * 2, kernel_size=3, padding=1, groups=hidden_dim * 2)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        x = self.project_in(x)
        x = self.dwconv(x)
        x1, x2 = x.chunk(2, dim=1)
        x = F.gelu(x1) * x2  # Gating
        x = self.project_out(x)
        return x

class TransformerBlock(nn.Module):
    """
    Super-Charged Restoration Block:
    Norm -> AdaWM -> Attn -> Add
    Norm -> AdaWM -> GDFN -> ChannelAttn -> Add
    """
    def __init__(self, dim, num_heads=6, window_size=WINDOW_SIZE, shift_size=0, time_dim=TIME_DIM):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, dim)
        self.adawm1 = AdaWM(dim, time_dim=time_dim)
        self.attn = WindowAttention(dim, num_heads, window_size, shift_size)
        
        self.norm2 = nn.GroupNorm(1, dim)
        self.adawm2 = AdaWM(dim, time_dim=time_dim)
        
        # IMPROVEMENT: GDFN instead of MLP
        self.ffn = GatedDconvFeedForward(dim)
        # IMPROVEMENT: Channel Attention
        self.cab = ChannelAttention(dim)

    def forward(self, x, time_emb):
        # 1. Attention Branch
        resid = x
        h = self.norm1(x)
        h = self.adawm1(h, time_emb)
        h = self.attn(h)
        x = resid + h
        
        # 2. Feed-Forward Branch (GDFN + CAB)
        resid = x
        h = self.norm2(x)
        h = self.adawm2(h, time_emb)
        h = self.ffn(h) # GDFN takes (B,C,H,W) directly
        h = self.cab(h) # Channel Attention
        
        x = resid + h
        return x

class SpectralEncoder(nn.Module):
    def __init__(self, in_channels=NUM_SPECTRAL_BANDS, out_channels=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        )
    def forward(self, x):
        return self.net(x)

# ===========================================================
# 5. Main Architecture
# ===========================================================
class DiTStage(nn.Module):
    def __init__(self, dim, num_blocks, mode='none'):
        super().__init__()
        blocks = []
        for i in range(num_blocks):
            shift = 0 if (i % 2 == 0) else (WINDOW_SIZE // 2)
            blocks.append(TransformerBlock(dim, shift_size=shift))
        self.blocks = nn.ModuleList(blocks)
        
        if mode == 'down':
            self.resample = nn.Sequential(nn.PixelUnshuffle(2), nn.Conv2d(dim * 4, dim, 1))
        elif mode == 'up':
            self.resample = nn.Sequential(nn.Conv2d(dim, dim * 4, 1), nn.PixelShuffle(2))
        else:
            self.resample = nn.Identity()

    def forward(self, x, time_emb):
        for block in self.blocks:
            x = block(x, time_emb)
        skip = x
        x = self.resample(x)
        return x, skip

class UNetDiT_SR(nn.Module):
    def __init__(self):
        super().__init__()
        self.cond_encoder = SpectralEncoder(in_channels=NUM_SPECTRAL_BANDS, out_channels=128)
        self.input_proj = nn.Conv2d(LATENT_CHANNELS + 128, BASE_CHANNELS, kernel_size=1)
        self.time_mlp = nn.Sequential(nn.Linear(TIME_DIM, TIME_DIM), nn.GELU(), nn.Linear(TIME_DIM, TIME_DIM))

        self.stage1 = DiTStage(BASE_CHANNELS, TRANSFORMER_BLOCKS[0], 'down')
        self.stage2 = DiTStage(BASE_CHANNELS, TRANSFORMER_BLOCKS[1], 'down')
        self.stage3 = DiTStage(BASE_CHANNELS, TRANSFORMER_BLOCKS[2], 'down')
        self.stage4 = DiTStage(BASE_CHANNELS, TRANSFORMER_BLOCKS[3], 'none')

        self.up3_resample = nn.Sequential(nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS*4, 1), nn.PixelShuffle(2))
        self.merge3 = nn.Conv2d(BASE_CHANNELS * 2, BASE_CHANNELS, 1)
        self.stage3_up = DiTStage(BASE_CHANNELS, TRANSFORMER_BLOCKS[2], 'none')

        self.up2_resample = nn.Sequential(nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS*4, 1), nn.PixelShuffle(2))
        self.merge2 = nn.Conv2d(BASE_CHANNELS * 2, BASE_CHANNELS, 1)
        self.stage2_up = DiTStage(BASE_CHANNELS, TRANSFORMER_BLOCKS[1], 'none')

        self.up1_resample = nn.Sequential(nn.Conv2d(BASE_CHANNELS, BASE_CHANNELS*4, 1), nn.PixelShuffle(2))
        self.merge1 = nn.Conv2d(BASE_CHANNELS * 2, BASE_CHANNELS, 1)
        self.stage1_up = DiTStage(BASE_CHANNELS, TRANSFORMER_BLOCKS[0], 'none')

        self.out_proj = nn.Conv2d(BASE_CHANNELS, LATENT_CHANNELS, 1)

    def forward(self, lr_multispectral, xt, t):
        cond_feat = self.cond_encoder(lr_multispectral)
        t_emb = timestep_embedding(t, TIME_DIM).to(xt.device)
        t_emb = self.time_mlp(t_emb)

        x = torch.cat([xt, cond_feat], dim=1) 
        x = self.input_proj(x)

        x, s1 = self.stage1(x, t_emb)
        x, s2 = self.stage2(x, t_emb)
        x, s3 = self.stage3(x, t_emb)
        x, _ = self.stage4(x, t_emb)

        x = self.up3_resample(x)
        x = torch.cat([x, s3], dim=1)
        x = self.merge3(x)
        x, _ = self.stage3_up(x, t_emb)

        x = self.up2_resample(x)
        x = torch.cat([x, s2], dim=1)
        x = self.merge2(x)
        x, _ = self.stage2_up(x, t_emb)

        x = self.up1_resample(x)
        x = torch.cat([x, s1], dim=1)
        x = self.merge1(x)
        x, _ = self.stage1_up(x, t_emb)

        return self.out_proj(x)
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
# ===========================================================
# 1. Configuration
# ===========================================================
# Paths
DATA_DIR = "/sr_data2/dit_sr_f4_multispectral"
TRAIN_LR_FILE = os.path.join(DATA_DIR, "train_LR_multispectral.npy")
TRAIN_HR_LAT_FILE = os.path.join(DATA_DIR, "train_HR_latents_2.npy")

# Training Hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 5e-5         
EPOCHS = 100                 # GDFN converges faster than MLP, but 100 is safe
NOISE_STEPS = 15             # ResShift steps
KAPPA = 2.0                  # ResShift Noise Variance

# RGB Indices for Sentinel-2 (Usually B4, B3, B2) to create the 3-channel residue
RGB_INDICES = [3, 2, 1] 

# ===========================================================
# 2. Wavelet Loss Function
# ===========================================================
def wavelet_loss(pred, target):
    """
    Calculates loss in the Wavelet Domain.
    Forces the model to learn structure (LL) and sharp textures (HF) separately.
    """
    # 1. Decompose -> [LL, HL, LH, HH]
    # Input: (B, 3, 24, 24) -> Output: (B, 12, 12, 12)
    pred_dwt = haar_dwt(pred)
    target_dwt = haar_dwt(target)
    
    # 2. Split Bands
    # First 3 channels are LL (Low Freq), rest are High Freq
    C = 3 
    
    pred_LL = pred_dwt[:, :C, :, :]
    target_LL = target_dwt[:, :C, :, :]
    
    pred_HF = pred_dwt[:, C:, :, :]
    target_HF = target_dwt[:, C:, :, :]
    
    # 3. Calculate Loss
    # Structure (LL) -> MSE (Smoothness)
    loss_structure = F.mse_loss(pred_LL, target_LL)
    
    # Texture (HF) -> L1 (Sparsity/Sharpness)
    loss_texture = F.l1_loss(pred_HF, target_HF)
    
    # Weight texture higher to improve LPIPS
    return loss_structure + 2.0 * loss_texture

# ===========================================================
# 3. Dataset & Scheduler
# ===========================================================
class SRLatentDataset(Dataset):
    def __init__(self, lr_path, hr_lat_path):
        print(f"Loading data from {lr_path}...")
        self.lr = np.load(lr_path)       
        self.hr_lat = np.load(hr_lat_path) 
        self.n = len(self.lr)
        print(f"Loaded {self.n} samples.")

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return torch.from_numpy(self.lr[idx]), torch.from_numpy(self.hr_lat[idx])

class ResShiftScheduler:
    def __init__(self, num_steps=15):
        self.num_steps = num_steps
        self.timesteps = torch.arange(1, num_steps + 1).float()
        self.etas = self.timesteps / num_steps

    def get_eta(self, t_idx):
        return self.etas[t_idx]

# ===========================================================
# 4. Training Loop
# ===========================================================
def train():
    # 1. Setup Data
    if not os.path.exists(TRAIN_LR_FILE):
        print(f"ERROR: Data file {TRAIN_LR_FILE} not found.")
        return

    train_ds = SRLatentDataset(TRAIN_LR_FILE, TRAIN_HR_LAT_FILE)
    loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    # 2. Setup Model (GDFN + ChannelAttn + AdaWM)
    model = UNetDiT_SR().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Calculate params (Should be slightly higher due to GDFN/CAB)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters (GDFN-SpecWave): {param_count/1e6:.2f}M")

    # 3. Scheduler
    scheduler = ResShiftScheduler(num_steps=NOISE_STEPS)

    print(f"Starting Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        model.train()
        pbar = tqdm(loader)
        loss_ema = None
        
        for lr, hr_lat in pbar:
            # Inputs
            lr, hr_lat = lr.to(DEVICE), hr_lat.to(DEVICE)
            B = lr.shape[0]
            
            # -----------------------------------------------------------
            # ResShift Setup
            # -----------------------------------------------------------
            # 1. Extract RGB from LR for the "Start Point" y0
            # lr is (B, 12, 24, 24), lr_rgb is (B, 3, 24, 24)
            lr_rgb = lr[:, RGB_INDICES, :, :]
            
            # 2. Sample Time
            t_idx = torch.randint(0, NOISE_STEPS, (B,), device=DEVICE)
            eta_t = scheduler.get_eta(t_idx.cpu()).to(DEVICE).view(B, 1, 1, 1)
            
            # 3. Create Noisy State xt
            residual = lr_rgb - hr_lat
            noise = torch.randn_like(hr_lat)
            std_dev = KAPPA * torch.sqrt(eta_t)
            
            xt = hr_lat + eta_t * residual + std_dev * noise
            
            # -----------------------------------------------------------
            # Optimization
            # -----------------------------------------------------------
            # 4. Predict
            # Model takes FULL 12-channel LR (via SpectralEncoder)
            pred_x0 = model(lr, xt, t_idx)
            
            # 5. Loss (Using Wavelet Loss)
            loss = wavelet_loss(pred_x0, hr_lat)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Clip Grads (GDFN can be sensitive to exploding grads initially)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            # Logging
            if loss_ema is None: loss_ema = loss.item()
            else: loss_ema = 0.95 * loss_ema + 0.05 * loss.item()
            
            pbar.set_description(f"Ep {epoch+1}/{EPOCHS} | Loss: {loss_ema:.4f}")

        # Checkpointing
        if (epoch + 1) % 5 == 0:
            save_path = f"dit_sr_gdfn_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            print(f"Saved checkpoint: {save_path}")

if __name__ == "__main__":
    train()
