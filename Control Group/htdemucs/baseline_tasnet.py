import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

class GlobalLayerNorm(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        return self.gamma * (x - mean) / (var + 1e-8).sqrt() + self.beta


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.PReLU(),
            GlobalLayerNorm(hidden_channels),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                      dilation=dilation, padding=padding, groups=hidden_channels),
            nn.PReLU(),
            GlobalLayerNorm(hidden_channels),
            nn.Conv1d(hidden_channels, in_channels, 1),
        )

    def forward(self, x):
        return x + self.conv(x)


class ConvTasNetBaseline(nn.Module):
    """
    純音訊 Conv-TasNet baseline，無任何視覺模組。
    超參數與 model_tasnet_refine.py 的 audio_model 完全一致，
    確保對比實驗的音訊主幹是相同的架構和容量。

    N=256, L=40, B=128, H=256, P=3, X=8, R=3
    """

    N = 256
    L = 40
    B = 128
    H = 256
    P = 3
    X = 8
    R = 3
    NUM_SOURCES = 2

    def __init__(self):
        super().__init__()

        self.encoder    = nn.Conv1d(1, self.N, kernel_size=self.L,
                                    stride=self.L // 2, bias=False)
        self.bottleneck = nn.Conv1d(self.N, self.B, 1)

        self.tcn_blocks = nn.ModuleList([
            DepthwiseSeparableConv(self.B, self.H, self.P, dilation=2 ** (i % self.X))
            for i in range(self.R * self.X)
        ])

        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(self.B, self.N * self.NUM_SOURCES, 1),
            nn.Sigmoid(),
        )

        self.decoder = nn.ConvTranspose1d(self.N, 1,
                                          kernel_size=self.L,
                                          stride=self.L // 2, bias=False)

    def forward(self, mix_wav):
        B, C, L_audio = mix_wav.shape

        enc = F.relu(self.encoder(mix_wav))
        x   = self.bottleneck(enc)

        for block in self.tcn_blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)

        masks = self.mask_net(x)
        masks = masks.view(B, self.NUM_SOURCES, self.N, enc.shape[-1])

        masked = enc * masks[:, 0]
        out    = self.decoder(masked)

        if out.shape[-1] > L_audio:
            out = out[..., :L_audio]
        elif out.shape[-1] < L_audio:
            out = F.pad(out, (0, L_audio - out.shape[-1]))

        return out


if __name__ == "__main__":
    model = ConvTasNetBaseline()
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total/1e6:.2f}M")
    print("✅ Baseline 架構正常")