import torch
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.unet import MelSpectrogramUNet, ResidualBlock, AttentionBlock

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create model
model = MelSpectrogramUNet(input_channels=1, base_channels=64).to(device)

# Input shape: (batch, channels, height, width) = (1, 1, 128, 216)
x = torch.randn(1, 1, 128, 216).to(device)
timestep = torch.randint(0, 1000, (1,)).to(device)

print(f"Input shape: {x.shape}")

# Manually trace through the network to debug
time_emb = model.time_mlp(timestep)
print(f"Time embedding shape: {time_emb.shape}")

x = model.init_conv(x)
print(f"After init_conv: {x.shape}")  # Should be (1, 64, 128, 216)

x1, skip1 = model.down1(x, time_emb)
print(f"After down1 - x: {x1.shape}, skip1: {skip1.shape}")

x2, skip2 = model.down2(x1, time_emb)
print(f"After down2 - x: {x2.shape}, skip2: {skip2.shape}")

x3, skip3 = model.down3(x2, time_emb)
print(f"After down3 - x: {x3.shape}, skip3: {skip3.shape}")

# Bottleneck
for layer in model.bottleneck:
    if isinstance(layer, ResidualBlock):
        x3 = layer(x3, time_emb)
    else:
        x3 = layer(x3)
print(f"After bottleneck: {x3.shape}")

print("\n--- DECODER ---")
print(f"up3 receives: x3 {x3.shape} and skip3 {skip3.shape}")
print(f"up3.upsample expects input of {model.up3.upsample.in_channels} channels")
print(f"up3 will upsample to {model.up3.upsample.out_channels} channels")
print(f"After concat with skip3: {model.up3.upsample.out_channels} + {skip3.shape[1]} = {model.up3.upsample.out_channels + skip3.shape[1]}")
print(f"up3.res_block1 expects input of {model.up3.res_block1.block1[2].in_channels} channels")

try:
    x = model.up3(x3, skip3, time_emb)
    print(f"✓ After up3: {x.shape}")
    
    print(f"\nup2 receives: x {x.shape} and skip2 {skip2.shape}")
    print(f"up2 will upsample to {model.up2.upsample.out_channels} channels")
    print(f"After concat with skip2: {model.up2.upsample.out_channels} + {skip2.shape[1]} = {model.up2.upsample.out_channels + skip2.shape[1]}")
    
    x = model.up2(x, skip2, time_emb)
    print(f"✓ After up2: {x.shape}")
    
    print(f"\nup1 receives: x {x.shape} and skip1 {skip1.shape}")
    print(f"up1 will upsample to {model.up1.upsample.out_channels} channels")
    print(f"After concat with skip1: {model.up1.upsample.out_channels} + {skip1.shape[1]} = {model.up1.upsample.out_channels + skip1.shape[1]}")
    
    x = model.up1(x, skip1, time_emb)
    print(f"✓ After up1: {x.shape}")
    
    x = model.final_conv(x)
    print(f"✓ After final_conv: {x.shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
