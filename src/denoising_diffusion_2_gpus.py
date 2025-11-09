import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import List
import random, math, os, sys
import numpy as np
from tqdm import tqdm
from timm.utils import ModelEmaV3
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

#root_directory = os.path.dirname(os.getcwd())
#sys.path.append(os.path.join(root_directory, 'src'))

from text_encoders import clip_model, bert_model
from dataloaders import caption_dataset

class SinusoidalEmbeddings(nn.Module):
    def __init__(self, time_steps:int, embed_dim:int):
        super().__init__()
        position = torch.arange(time_steps).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, embed_dim, 2).float() * -(math.log(10000.0) / embed_dim))
        embeddings = torch.zeros(time_steps, embed_dim, requires_grad=False)
        embeddings[:, 0::2] = torch.sin(position * div)
        embeddings[:, 1::2] = torch.cos(position * div)
        self.embeddings = embeddings

    def forward(self, x, t):
        embeds = self.embeddings.to(t.device)[t]
        return embeds[:, :, None, None]
    

class ResBlock(nn.Module):
    def __init__(self, C:int, num_groups:int, dropout_prob:float):
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.gnorm1 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.gnorm2 = nn.GroupNorm(num_groups=num_groups, num_channels=C)
        self.conv1 = nn.Conv2d(C, C, 3, padding=1)
        self.conv2 = nn.Conv2d(C, C, 3, padding=1)
        self.dropout = nn.Dropout(p=dropout_prob, inplace=True)

    def forward(self, x, embeddings):
        x = x + embeddings[:, :x.shape[1], :, :]
        r = self.conv1(self.relu(self.gnorm1(x)))
        r = self.dropout(r)
        r = self.conv2(self.relu(self.gnorm2(r)))
        return r + x
    

class Attention(nn.Module):
    def __init__(self, C:int, num_heads:int, dropout_prob:float):
        super().__init__()
        self.proj1 = nn.Linear(C, C*3)
        self.proj2 = nn.Linear(C, C)
        self.num_heads = num_heads
        self.dropout_prob = dropout_prob

    def forward(self, x):
        h, w = x.shape[2:]
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.proj1(x)
        x = rearrange(x, 'b L (C H K) -> K b H L C', K=3, H=self.num_heads)
        q, k, v = x[0], x[1], x[2]
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout_prob)
        x = rearrange(x, 'b H (h w) C -> b h w (C H)', h=h, w=w)
        x = self.proj2(x)
        return rearrange(x, 'b h w C -> b C h w')
    
class UnetLayer(nn.Module):
    def __init__(self, upscale, attention, num_groups, dropout_prob, num_heads, C):
        super().__init__()
        self.ResBlock1 = ResBlock(C, num_groups, dropout_prob)
        self.ResBlock2 = ResBlock(C, num_groups, dropout_prob)
        if upscale:
            self.conv = nn.ConvTranspose2d(C, C//2, 4, stride=2, padding=1)
        else:
            self.conv = nn.Conv2d(C, C*2, 3, stride=2, padding=1)
        if attention:
            self.attention_layer = Attention(C, num_heads, dropout_prob)

    def forward(self, x, embeddings):
        x = self.ResBlock1(x, embeddings)
        if hasattr(self, 'attention_layer'):
            x = self.attention_layer(x)
        x = self.ResBlock2(x, embeddings)
        return self.conv(x), x
    
class UNET(nn.Module):
    def __init__(self,
                 Channels=[64,128,256,512,512,384],
                 Attentions=[False,True,False,False,False,True],
                 Upscales=[False,False,False,True,True,True],
                 num_groups=32,
                 dropout_prob=0.1,
                 num_heads=8,
                 input_channels=1,
                 output_channels=1,
                 time_steps=1000,
                 text_embed_dim=512):
        super().__init__()
        self.num_layers = len(Channels)
        self.shallow_conv = nn.Conv2d(input_channels, Channels[0], 3, padding=1)
        self.late_conv = nn.Conv2d((Channels[-1]//2)+Channels[0], (Channels[-1]//2), 3, padding=1)
        self.output_conv = nn.Conv2d((Channels[-1]//2), output_channels, 1)
        self.relu = nn.ReLU(inplace=True)

        self.embeddings = SinusoidalEmbeddings(time_steps, max(Channels))
        self.text_proj = nn.Linear(text_embed_dim, max(Channels))

        for i in range(self.num_layers):
            layer = UnetLayer(
                upscale=Upscales[i],
                attention=Attentions[i],
                num_groups=num_groups,
                dropout_prob=dropout_prob,
                C=Channels[i],
                num_heads=num_heads
            )
            setattr(self, f'Layer{i+1}', layer)

    def forward(self, x, t, text_emb):
        x = self.shallow_conv(x)
        residuals = []

        text_emb = self.text_proj(text_emb).unsqueeze(-1).unsqueeze(-1)
        time_emb = self.embeddings(x, t)
        combined_emb = time_emb + text_emb

        for i in range(self.num_layers // 2):
            layer = getattr(self, f'Layer{i+1}')
            x, r = layer(x, combined_emb)
            residuals.append(r)

        for i in range(self.num_layers // 2, self.num_layers):
            layer = getattr(self, f'Layer{i+1}')
            x = torch.concat((layer(x, combined_emb)[0], residuals[self.num_layers - i - 1]), dim=1)

        return self.output_conv(self.relu(self.late_conv(x)))
    

class DDPM_Scheduler(nn.Module):
    def __init__(self, num_time_steps=1000):
        super().__init__()
        self.beta = torch.linspace(1e-4, 0.02, num_time_steps, requires_grad=False)
        alpha = 1 - self.beta
        self.alpha = torch.cumprod(alpha, dim=0).requires_grad_(False)

    def forward(self, t):
        device = t.device
        return self.beta.to(device)[t], self.alpha.to(device)[t]
    
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

dist.init_process_group("nccl")

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device("cuda", local_rank)

distributed = True
dataloader = caption_dataset()
train_dataloader, train_sampler = dataloader.get_dataloader(partition="train", batch_size=16, distributed=distributed)

scheduler = DDPM_Scheduler(num_time_steps=1000)

model = UNET(
    input_channels=3,
    output_channels=3,
    text_embed_dim=train_dataloader.dataset.embed_dim
).to(device)

model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# Optimizer and loss (same)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
criterion = nn.MSELoss()

num_epochs = 15
for epoch in range(num_epochs):
    if train_sampler is not None:
        train_sampler.set_epoch(epoch)
    model.train()
    total_loss = 0
    for imgs, text_emb in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, text_emb = imgs.to(device), text_emb.to(device)
        b = imgs.size(0)

        # sample timesteps
        t = torch.randint(0, 1000, (b,), device=device)
        e = torch.randn_like(imgs)

        beta_t, alpha_t = scheduler(t)
        a = alpha_t.view(b, 1, 1, 1)
        noisy_imgs = (torch.sqrt(a) * imgs) + (torch.sqrt(1 - a) * e)

        pred_noise = model(noisy_imgs, t, text_emb)
        loss = criterion(pred_noise, e)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        del loss, pred_noise, noisy_imgs, e, t, beta_t, alpha_t, a
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if dist.get_rank() == 0:
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_dataloader):.4f}")

