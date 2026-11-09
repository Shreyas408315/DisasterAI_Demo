import torch
from torch import nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    """Vanilla U-Net baseline; all parameters randomly initialized."""
    def __init__(self, in_ch=2, out_ch=1, base=16):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base*2)
        self.e3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*4, base*8)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = ConvBlock(base*8, base*4)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = ConvBlock(base*4, base*2)
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = ConvBlock(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        d = self.d3(torch.cat([self.up3(z), c], dim=1))
        d = self.d2(torch.cat([self.up2(d), b], dim=1))
        d = self.d1(torch.cat([self.up1(d), a], dim=1))
        return self.out(d)

class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.gate = nn.Sequential(nn.Conv2d(gate_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.skip = nn.Sequential(nn.Conv2d(skip_ch, inter_ch, 1), nn.BatchNorm2d(inter_ch))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(inter_ch, 1, 1), nn.Sigmoid())

    def forward(self, gate, skip):
        gate_proj = self.gate(gate)
        skip_proj = self.skip(skip)
        # Decoder gating features are lower-resolution than encoder skip features.
        if gate_proj.shape[-2:] != skip_proj.shape[-2:]:
            gate_proj = F.interpolate(
                gate_proj, size=skip_proj.shape[-2:],
                mode="bilinear", align_corners=False
            )
        alpha = self.psi(gate_proj + skip_proj)
        return skip * alpha

class ProposedAttentionUNet(nn.Module):
    """Proposed attention-gated U-Net. Random initialization only."""
    def __init__(self, in_ch=2, out_ch=1, base=16):
        super().__init__()
        self.e1 = ConvBlock(in_ch, base)
        self.e2 = ConvBlock(base, base*2)
        self.e3 = ConvBlock(base*2, base*4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(base*4, base*8)
        self.ag3 = AttentionGate(base*8, base*4, base*2)
        self.up3 = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.d3 = ConvBlock(base*8, base*4)
        self.ag2 = AttentionGate(base*4, base*2, base)
        self.up2 = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.d2 = ConvBlock(base*4, base*2)
        self.ag1 = AttentionGate(base*2, base, max(4, base//2))
        self.up1 = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.d1 = ConvBlock(base*2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        a = self.e1(x)
        b = self.e2(self.pool(a))
        c = self.e3(self.pool(b))
        z = self.bottleneck(self.pool(c))
        u3 = self.up3(z)
        c_att = self.ag3(z, c)
        d = self.d3(torch.cat([u3, c_att], dim=1))
        u2 = self.up2(d)
        b_att = self.ag2(d, b)
        d = self.d2(torch.cat([u2, b_att], dim=1))
        u1 = self.up1(d)
        a_att = self.ag1(d, a)
        d = self.d1(torch.cat([u1, a_att], dim=1))
        return self.out(d)
