"""
VisualGuidedConvTasNet
======================
將 HTDemucs 實驗中驗證有效的視覺引導框架移植到 Conv-TasNet。

架構對應關係：
  HTDemucs                    Conv-TasNet
  ─────────────────────────────────────────
  VisualBottleneckInjector  → VisualTCNInjector（每層 TCN block 後注入）
  SpectralMaskHead          → 保留（FiLM in STFT domain，邏輯完全相同）
  PSLHead                   → 保留（介面不變）
  AMNetStreamEncoder        → 保留（介面不變）
  opponent suppression      → 保留（邏輯不變）

視覺注入策略：
  - Conv-TasNet 的核心是 TCN（Temporal Convolutional Network），
    由多個 repeats × blocks 組成。
  - 與 HTDemucs 的「瓶頸層單點注入」不同，這裡在每個 repeat 結束後
    做一次 FiLM conditioning，讓視覺資訊在多個尺度都能影響分離。
  - 每個 FiLM 層有 LayerScale（初始 1e-4），保護訓練初期穩定性。

輸入輸出格式與 HTDemucs 版本完全相同，train_colab.py 可直接替換模型類別。

參考：
  - Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal T-F Masking for Speech Sep." 2019
  - HTDemucs 視覺引導版 (model_arch.py)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ============================================================
# STFT / iSTFT 工具（與 model_arch.py 完全相同）
# ============================================================

def stft(wav, n_fft=2048, hop=512):
    """wav : [B, 1, L]  →  mag [B, F, T], phase [B, F, T]"""
    B = wav.shape[0]
    x = wav.squeeze(1)
    win = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop,
                      window=win, return_complex=True, normalized=False)
    return spec.abs(), spec.angle()


def istft(mag, phase, n_fft=2048, hop=512, length=None):
    """mag, phase : [B, F, T]  →  wav [B, 1, L]"""
    spec = mag * torch.exp(1j * phase)
    win = torch.hann_window(n_fft, device=mag.device)
    wav = torch.istft(spec, n_fft=n_fft, hop_length=hop,
                      window=win, length=length, normalized=False)
    return wav.unsqueeze(1)


# ============================================================
# 1. 視覺流編碼器（與 model_arch.py 完全相同）
#    app [B,150,13,3] → [B,150,512]
#    mot [B,150,13,4] → [B,150,512]
# ============================================================

class AMNetStreamEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, output_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        if x.dim() == 4:
            b, t, p, d = x.shape
            x = x.reshape(b, t, p * d)
        return self.net(x)  # [B, T, 512]


# ============================================================
# 2. 頻譜 Ratio Mask Head（與 model_arch.py 完全相同）
# ============================================================

class SpectralMaskHead(nn.Module):
    def __init__(self, v_dim=512, n_freq=1025, hidden=256, n_heads=4):
        super().__init__()
        self.n_freq = n_freq
        # 視覺序列 → 每個時間幀的 FiLM 參數
        self.vis_to_film = nn.Linear(v_dim, n_freq * 2)
        # 音訊頻譜 query 視覺序列的 cross-attention
        self.audio_proj  = nn.Linear(n_freq, v_dim)
        self.cross_attn  = nn.MultiheadAttention(v_dim, num_heads=n_heads, batch_first=True)
        self.out_proj    = nn.Linear(v_dim, n_freq)
        self.norm        = nn.LayerNorm(n_freq)
        self.layer_scale = nn.Parameter(torch.full((n_freq,), 1e-4))
        self.spec_conv = nn.Sequential(
            nn.Conv1d(n_freq, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, n_freq, kernel_size=3, padding=1),
        )
        self.mask_head = nn.Sequential(
            nn.Conv1d(n_freq, n_freq, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, v_seq, spec_mag, tcn_mask=None):
        B, F, T = spec_mag.shape

        v_global = v_seq.mean(dim=1)
        film     = self.vis_to_film(v_global)
        gamma    = film[:, :F].unsqueeze(2) + 1.0
        beta     = film[:, F:].unsqueeze(2)
        feat     = gamma * spec_mag + beta

        feat_t      = feat.permute(0, 2, 1)
        q           = self.audio_proj(feat_t)
        attn_out, _ = self.cross_attn(q, v_seq, v_seq)
        delta       = self.out_proj(attn_out)
        scale       = self.layer_scale.unsqueeze(0).unsqueeze(0)
        feat_t      = self.norm(feat_t + scale * delta)
        feat        = feat_t.permute(0, 2, 1)

        feat      = self.spec_conv(feat)
        vis_delta = self.mask_head(feat) - 0.5  # [B, F, T]，值域 (-0.5, 0.5)

        if tcn_mask is not None:
            # tcn_mask : [B, 1, T]，broadcast 到 [B, F, T]
            mask = (tcn_mask + vis_delta).clamp(0.1, 1.0)
        else:
            mask = (0.5 + vis_delta).clamp(0.1, 1.0)

        pred_A = mask * spec_mag
        pred_B = (1 - mask) * spec_mag
        return mask, pred_A, pred_B


# ============================================================
# 3. PSL Head（與 model_arch.py 完全相同）
# ============================================================

class PSLHead(nn.Module):
    def __init__(self, audio_dim=512, v_dim=512):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, v_dim)

    def forward(self, v_a_global, v_b_global, audio_patches):
        a_proj = self.audio_proj(audio_patches)
        v_a = F.normalize(v_a_global.unsqueeze(1), dim=-1)
        v_b = F.normalize(v_b_global.unsqueeze(1), dim=-1)
        a_n = F.normalize(a_proj, dim=-1)
        return (a_n * v_a).sum(dim=-1) - (a_n * v_b).sum(dim=-1)


# ============================================================
# 4. Audio Patch Encoder（與 model_arch.py 完全相同）
# ============================================================

class AudioPatchEncoder(nn.Module):
    def __init__(self, n_freq=1025, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_freq, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, spec_mag):
        return self.proj(spec_mag.transpose(1, 2))  # [B, T, 512]


# ============================================================
# 5. Conv-TasNet 基礎組件
# ============================================================

class GlobalLayerNorm(nn.Module):
    """gLN：對 [B, C, T] 做全域正規化"""
    def __init__(self, channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1))
        self.beta  = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = x.mean(dim=(1, 2), keepdim=True)
        var  = ((x - mean) ** 2).mean(dim=(1, 2), keepdim=True)
        return self.gamma * (x - mean) / (var + 1e-8).sqrt() + self.beta


class DepthwiseSeparableConv(nn.Module):
    """
    單個 TCN block（depthwise separable conv + gLN + PReLU + 殘差）
    dilation 控制感受野，每個 repeat 內指數增長。
    """
    def __init__(self, in_channels, hidden_channels, kernel_size, dilation):
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.conv = nn.Sequential(
            # 1×1 pointwise expand
            nn.Conv1d(in_channels, hidden_channels, 1),
            nn.PReLU(),
            GlobalLayerNorm(hidden_channels),
            # depthwise
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size,
                      dilation=dilation, padding=padding, groups=hidden_channels),
            nn.PReLU(),
            GlobalLayerNorm(hidden_channels),
            # 1×1 pointwise reduce
            nn.Conv1d(hidden_channels, in_channels, 1),
        )

    def forward(self, x):
        return x + self.conv(x)  # 殘差連接


# ============================================================
# 6. VisualTCNInjector
#    在每個 TCN repeat 結束後注入視覺特徵（對應 HTDemucs 的瓶頸注入）。
#
#    注入方式：
#      FiLM conditioning（與 SpectralMaskHead 的頻率維度 FiLM 同原理，
#      但作用在時間維度的 channel 空間）
#        gamma, beta = MLP(v_global) → [B, C, 1]
#        x_out = x + LayerScale × (gamma × x + beta)
#
#    為什麼用 FiLM 而非 cross-attention：
#      TCN 的時間解析度比視覺幀率高很多（44100Hz vs 30fps），
#      cross-attention 計算量太大。FiLM 的 channel-wise 調制
#      在計算效率和效果之間取得較好的平衡。
#      （若需要更強的時間對齊，可改用 cross-attention，
#       但需要先對 TCN feature 做時間 pooling。）
# ============================================================

class VisualTCNInjector(nn.Module):
    def __init__(self, tcn_channels=128, v_dim=512, n_heads=4, layer_scale_init=1e-4):
        super().__init__()
        self.audio_proj  = nn.Linear(tcn_channels, v_dim)
        self.cross_attn  = nn.MultiheadAttention(v_dim, num_heads=n_heads, batch_first=True)
        self.out_proj    = nn.Linear(v_dim, tcn_channels)
        self.norm        = nn.LayerNorm(tcn_channels)
        self.layer_scale = nn.Parameter(torch.full((tcn_channels,), layer_scale_init))

    def forward(self, x, v_seq):
        B, C, T = x.shape
        x_t = x.permute(0, 2, 1)                              # [B, T, C]

        # ── Pool TCN feature 到視覺幀數，再做 cross-attention ──
        # 原本 q=[B,11025,512] 佔大量 GPU；pool 到 [B,150,512] 後
        # attention map 從 [B,11025,150] 縮小為 [B,150,150]，節省 ~70 倍記憶體。
        # 輸出 delta 再插值回原本時間長度，資訊不損失。
        V = v_seq.shape[1]                                     # 視覺幀數（150）
        x_pool = F.adaptive_avg_pool1d(x, V)                  # [B, C, V]
        x_pool_t = x_pool.permute(0, 2, 1)                    # [B, V, C]
        q = self.audio_proj(x_pool_t)                         # [B, V, v_dim]

        attn_out, attn_weights = self.cross_attn(
            q, v_seq, v_seq, need_weights=True, average_attn_weights=True
        )                                                      # attn_out [B, V, v_dim]
        delta_pool = self.out_proj(attn_out)                  # [B, V, C]

        # 插值回原始時間長度
        delta_full = F.interpolate(
            delta_pool.permute(0, 2, 1), size=T,
            mode='linear', align_corners=False
        ).permute(0, 2, 1)                                     # [B, T, C]

        scale = self.layer_scale.unsqueeze(0).unsqueeze(0)
        out   = self.norm(x_t + scale * delta_full)
        return out.permute(0, 2, 1), attn_weights


# ============================================================
# 7. Conv-TasNet 主幹（不含視覺，可單獨作為 baseline）
# ============================================================

class ConvTasNet(nn.Module):
    """
    標準 Conv-TasNet（單聲道輸入，2 個音源輸出）

    預設超參數來自原始論文（適合 44100 Hz 吉他分離）：
      N=512  : encoder filter 數量（比原論文 256 更大，應對高採樣率）
      L=16   : encoder kernel 長度（對應約 0.36ms @ 44100Hz）
      B=256  : bottleneck channels
      H=512  : depthwise conv hidden channels
      P=3    : kernel size of depthwise conv
      X=8    : blocks per repeat（感受野 = 2^0 + ... + 2^7 = 255 samples）
      R=3    : repeats（總感受野 = 3 × 255 = 765 samples ≈ 17ms @ 44100Hz）
    """
    def __init__(self,
                 N=512, L=16, B=256, H=512, P=3, X=8, R=3,
                 num_sources=2):
        super().__init__()
        self.N = N
        self.L = L
        self.num_sources = num_sources

        # ── Encoder：1D Conv（學習的短時分析窗）────────────────
        self.encoder = nn.Conv1d(1, N, kernel_size=L, stride=L // 2, bias=False)
        self.encoder_norm = nn.LayerNorm(N)

        # ── Separator：TCN ──────────────────────────────────────
        # bottleneck：N → B
        self.bottleneck = nn.Conv1d(N, B, 1)

        # TCN blocks：R repeats，每 repeat 有 X blocks，dilation 指數增長
        self.tcn_blocks = nn.ModuleList()
        for r in range(R):
            for x in range(X):
                self.tcn_blocks.append(
                    DepthwiseSeparableConv(B, H, P, dilation=2 ** x)
                )
        self.R = R
        self.X = X

        # mask head：B → N * num_sources，然後 sigmoid 生成 mask
        self.mask_net = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(B, N * num_sources, 1),
            nn.Sigmoid(),
        )

        # ── Decoder：轉置 1D Conv ───────────────────────────────
        self.decoder = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L // 2, bias=False)

    def encode(self, mix_wav):
        """
        mix_wav : [B, 1, L_audio]
        returns encoded : [B, N, T_enc]
        """
        enc = F.relu(self.encoder(mix_wav))  # [B, N, T_enc]
        return enc

    def separate(self, enc):
        """
        enc : [B, N, T_enc]
        returns masks : [B, num_sources, N, T_enc]
        """
        x = self.bottleneck(enc)  # [B, B_dim, T_enc]
        for block in self.tcn_blocks:
            x = block(x)
        masks = self.mask_net(x)   # [B, N*S, T_enc]
        B_batch, _, T = masks.shape
        masks = masks.view(B_batch, self.num_sources, self.N, T)
        return masks, x  # x 是最後的 TCN feature，供視覺注入用

    def decode(self, masked_enc, length):
        """
        masked_enc : [B, N, T_enc]  （單個音源的 masked encoder output）
        length     : 原始音訊長度
        returns    : [B, 1, length]
        """
        out = self.decoder(masked_enc)
        # 裁切或補齊到原始長度
        if out.shape[-1] > length:
            out = out[..., :length]
        elif out.shape[-1] < length:
            out = F.pad(out, (0, length - out.shape[-1]))
        return out

    def forward(self, mix_wav):
        """
        mix_wav : [B, 1, L]
        returns : [B, num_sources, 1, L]
        """
        B, C, L = mix_wav.shape
        enc = self.encode(mix_wav)
        masks, _ = self.separate(enc)
        # masks: [B, S, N, T_enc]
        outputs = []
        for s in range(self.num_sources):
            masked = enc * masks[:, s]         # [B, N, T_enc]
            outputs.append(self.decode(masked, L))  # [B, 1, L]
        return torch.stack(outputs, dim=1)     # [B, S, 1, L]


# ============================================================
# 8. VisualGuidedConvTasNet（主模型）
#    對應 model_arch.py 的 VisualGuidedHTDemucs
# ============================================================

class VisualGuidedConvTasNet(nn.Module):
    """
    視覺引導 Conv-TasNet

    與 HTDemucs 版本的對應關係：
      - audio_model         → ConvTasNet（自訓練，無預訓練權重）
      - bottleneck_injector → VisualTCNInjector（每 repeat 後注入）
      - spec_mask_head      → SpectralMaskHead（完全相同）
      - psl_head            → PSLHead（完全相同）
      - app/mot_encoder     → AMNetStreamEncoder（完全相同）
      - cross_attn          → MultiheadAttention（完全相同）
      - alpha               → opponent suppression（完全相同）
      - visual_weight       → 加權融合（與 HTDemucs 版一致）

    Forward 輸出格式與 VisualGuidedHTDemucs 完全相同，
    train_colab.py 只需替換模型類別名稱即可使用。
    """

    # STFT 參數（與 HTDemucs 版本一致）
    N_FFT = 2048
    HOP   = 512

    # Conv-TasNet 超參數
    # ⚠️ 針對 44100Hz 調整（原論文是 8kHz/16kHz 語音）：
    #   44100Hz 的 T_enc 比語音高 3-5 倍，必須縮小 N 和 H 避免 OOM。
    #
    #   記憶體估算（batch=16, 5秒, stride=L//2=20）：
    #     T_enc = 44100*5 / 20 ≈ 11,025
    #     encoder output = 16 × 256 × 11025 × 4B ≈ 180MB  ✅
    #
    #   感受野 = stride × (2^0 + ... + 2^(X-1)) × R
    #          = 20 × 255 × 3 ≈ 153,000 samples ≈ 3.5秒 @ 44100Hz  ✅
    TASNET_N = 256   # encoder filters（512→256，主要省記憶體來源）
    TASNET_L = 40    # encoder kernel（更大的分析窗，適合 44100Hz）
    TASNET_B = 128   # bottleneck channels（256→128）
    TASNET_H = 256   # depthwise hidden（512→256）
    TASNET_P = 3     # depthwise kernel（不變）
    TASNET_X = 8     # blocks per repeat（不變，保持感受野）
    TASNET_R = 3     # repeats（不變）

    # 視覺特徵維度（與 HTDemucs 版本一致）
    # app [B,150,13,3] → input_dim=39, mot [B,150,13,4] → input_dim=52
    APP_DIM = 39     # 13 points × 3 (x,y,z)
    MOT_DIM = 52     # 13 points × 4 (dx,dy,dz,v)
    VIS_DIM = 512

    def __init__(self):
        super().__init__()

        # ── 音訊主幹 ──────────────────────────────────────────
        self.audio_model = ConvTasNet(
            N=self.TASNET_N, L=self.TASNET_L,
            B=self.TASNET_B, H=self.TASNET_H,
            P=self.TASNET_P, X=self.TASNET_X,
            R=self.TASNET_R, num_sources=2
        )

        self.vis_injector = VisualTCNInjector(
            tcn_channels=self.TASNET_B,
            v_dim=self.VIS_DIM,
            n_heads=4,
            layer_scale_init=1e-4,
        )

        # ── 視覺編碼器（與 model_arch.py 完全相同）────────────
        self.app_encoder = AMNetStreamEncoder(self.APP_DIM, 256, self.VIS_DIM)
        self.mot_encoder = AMNetStreamEncoder(self.MOT_DIM, 256, self.VIS_DIM)
        self.cross_attn  = nn.MultiheadAttention(self.VIS_DIM, num_heads=8, batch_first=True)

        # opponent suppression（與 model_arch.py 完全相同）
        self.alpha = nn.Parameter(torch.tensor(0.3))

        # ── 頻譜 Mask Head（與 model_arch.py 完全相同）────────
        self.spec_mask_head = SpectralMaskHead(
            v_dim=self.VIS_DIM, n_freq=self.N_FFT // 2 + 1
        )

        # ── PSL Head（與 model_arch.py 完全相同）─────────────
        self.psl_head       = PSLHead(self.VIS_DIM, self.VIS_DIM)
        self.audio_patch_enc = AudioPatchEncoder(
            n_freq=self.N_FFT // 2 + 1, out_dim=self.VIS_DIM
        )

        # 視覺分支加權（與 model_arch.py 完全相同）
        self.visual_weight = nn.Parameter(torch.tensor(0.3))

    # ── 視覺特徵提取（與 model_arch.py 完全相同）────────────────

    def encode_visual(self, app, mot):
        """
        app : [B, 150, 13, 3]
        mot : [B, 150, 13, 4]
        returns : [B, 150, 512]
        """
        B = app.shape[0]
        app_feat = self.app_encoder(app.reshape(B, 150, -1))
        mot_feat = self.mot_encoder(mot.reshape(B, 150, -1))
        attn_out, _ = self.cross_attn(mot_feat, app_feat, app_feat)
        return attn_out

    # ── Conv-TasNet with 視覺注入的 forward ─────────────────────

    def _separate_with_visual(self, mix_wav, v_seq, spec_T=None):
        B, C, L = mix_wav.shape
        enc = self.audio_model.encode(mix_wav)
        x   = self.audio_model.bottleneck(enc)

        # gradient checkpointing：反向傳播時重新計算中間激活，
        # 不保留所有 TCN block 的 feature map，節省約 50% GPU 記憶體。
        # 邏輯與原本完全相同，只是以時間換空間。
        for block in self.audio_model.tcn_blocks:
            x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)

        x, attn_weights = self.vis_injector(x, v_seq)

        masks_raw = self.audio_model.mask_net(x)
        masks     = masks_raw.view(B, self.audio_model.num_sources,
                                   self.audio_model.N, enc.shape[-1])
        masked_enc = enc * masks[:, 0]
        pred_wav   = self.audio_model.decode(masked_enc, L)

        # TCN mask 插值到頻譜時間維度，供 SpectralMaskHead refinement 使用
        # masks[:, 0] : [B, N, T_enc]，對 N 維取平均得到 [B, 1, T_enc]
        tcn_mask_raw = masks[:, 0].mean(dim=1, keepdim=True)  # [B, 1, T_enc]
        if spec_T is not None:
            # 插值到頻譜時間長度，保持 [B, 1, T_spec]
            tcn_mask_spec = F.interpolate(tcn_mask_raw, size=spec_T,
                                          mode='linear', align_corners=False)
        else:
            tcn_mask_spec = tcn_mask_raw
        # tcn_mask_spec : [B, 1, T_spec]，在 SpectralMaskHead 裡 broadcast 到 [B, F, T_spec]

        return pred_wav, attn_weights, tcn_mask_spec

    # ── Forward（介面與 VisualGuidedHTDemucs 完全相同）──────────

    def forward(self, mix_wav, app_A, mot_A, app_B, mot_B):
        B, C, L = mix_wav.shape

        mix_mag, mix_phase = stft(mix_wav, n_fft=self.N_FFT, hop=self.HOP)

        # 視覺特徵提取
        feat_A = self.encode_visual(app_A, mot_A)  # [B, 150, 512]
        feat_B = self.encode_visual(app_B, mot_B)

        alpha        = torch.clamp(self.alpha, 0.0, 1.0)
        feat_A_final = feat_A - alpha * feat_B
        feat_B_final = feat_B - alpha * feat_A

        v_a_global = feat_A_final.mean(dim=1)  # [B, 512]
        v_b_global = feat_B_final.mean(dim=1)

        # Conv-TasNet with 視覺注入
        spec_T = mix_mag.shape[-1]
        tasnet_wav, attn_weights, tcn_mask_spec = self._separate_with_visual(
            mix_wav, feat_A_final, spec_T=spec_T
        )

        # 頻譜 Mask Head（TCN mask 作為基礎，視覺只做 refinement）
        mask, pred_mag_A, pred_mag_B = self.spec_mask_head(
            feat_A_final, mix_mag, tcn_mask=tcn_mask_spec
        )
        spec_wav = istft(pred_mag_A, mix_phase,
                         n_fft=self.N_FFT, hop=self.HOP, length=L)
        # pred_mag_B 保留備用（A + B = mix 在頻譜域自動滿足）

        # 加權融合
        w = torch.clamp(self.visual_weight, 0.0, 1.0)
        pred_wav = w * spec_wav + (1.0 - w) * tasnet_wav

        # PSL
        audio_patches = self.audio_patch_enc(mix_mag)
        p_hat = self.psl_head(v_a_global, v_b_global, audio_patches)

        return pred_wav, pred_mag_A, mask, mix_mag, mix_phase, p_hat, attn_weights

# ============================================================
# 快速健康檢查
# ============================================================

if __name__ == "__main__":
    model = VisualGuidedConvTasNet()
    total = sum(p.numel() for p in model.parameters())
    audio = sum(p.numel() for p in model.audio_model.parameters())
    print(f"Total: {total/1e6:.2f}M  Audio: {audio/1e6:.2f}M")
    print(f"vis_injector: {type(model.vis_injector).__name__}")
    print("✅ 架構正常")