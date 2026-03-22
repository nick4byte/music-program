import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from demucs.htdemucs import HTDemucs


# ============================================================
# STFT / iSTFT 工具
# ============================================================

def stft(wav, n_fft=2048, hop=512):
    """
    wav : [B, 1, L]
    returns mag [B, F, T], phase [B, F, T]  (F = n_fft//2+1)
    """
    B = wav.shape[0]
    x = wav.squeeze(1)  # [B, L]
    win = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop,
                      window=win, return_complex=True, normalized=False)
    mag   = spec.abs()            # [B, F, T]
    phase = spec.angle()          # [B, F, T]
    return mag, phase


def istft(mag, phase, n_fft=2048, hop=512, length=None):
    """
    mag, phase : [B, F, T]
    returns wav : [B, 1, L]
    """
    spec = mag * torch.exp(1j * phase)
    win = torch.hann_window(n_fft, device=mag.device)
    wav = torch.istft(spec, n_fft=n_fft, hop_length=hop,
                      window=win, length=length, normalized=False)
    return wav.unsqueeze(1)       # [B, 1, L]


# ============================================================
# 1. 視覺流編碼器 (AM-Net 精神)
#    app : [B, 150, 26, 3] → flatten → [B, 150, 78]
#    mot : [B, 150, 26, 4] → flatten → [B, 150, 104]
#    (data_loader 中 app = m[:,:,:3], mot = m[:,:,3:7]
#     其中 m 的 landmark 維度是 26，所以 flatten 後分別是 78, 104)
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
        # x: [B, T, P, D] 或 [B, T, P*D]
        if x.dim() == 4:
            b, t, p, d = x.shape
            x = x.reshape(b, t, p * d)
        return self.net(x)  # [B, T, 512]


# ============================================================
# 2. 頻譜 Ratio Mask Head（核心修改）
#    視覺特徵 [B,150,512] → 頻譜 mask [B,F,T_spec]
#
#    設計來源：
#    - AMnet (Zhu & Rahtu 2022)：appearance feature × audio spec feature → mask
#    - AVPC (Song & Zhang 2023)：visual feature → predict audio, mask × Xmix
#    - Wang et al. PSL：視覺引導頻譜分離
#
#    流程：
#      1. 視覺全局向量 v_global [B,512]
#      2. 混音頻譜 Xmix [B,F,T_spec] 展平後投影到 512 維
#      3. 視覺特徵作為 query，頻譜特徵作為 key/value → cross-attention
#      4. 投影回 [B,F,T_spec]，Sigmoid → ratio mask in (0,1)
#      5. pred_spec = mask × Xmix
# ============================================================

class SpectralMaskHead(nn.Module):
    def __init__(self, v_dim=512, n_freq=1025, hidden=256):
        """
        n_freq : n_fft//2 + 1，對應 STFT 的頻率 bin 數量
                 n_fft=2048 → n_freq=1025
        """
        super().__init__()
        self.n_freq = n_freq

        # 視覺全局向量 → 頻率軸的 FiLM 參數（在頻譜域做 scale/shift，安全）
        self.vis_to_film = nn.Sequential(
            nn.Linear(v_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_freq * 2),  # gamma + beta，每個頻率 bin 各一個
        )

        # 頻譜 feature extractor（在頻率軸做 1D conv）
        # 注意：Conv1d 輸出 [B, C, T]，不能用 LayerNorm(C)（它期待最後一維=C）
        # 改用 GroupNorm(num_groups=1, num_channels=hidden)，等價於 InstanceNorm，
        # 對 [B, C, T] 是安全的。
        self.spec_conv = nn.Sequential(
            nn.Conv1d(n_freq, hidden, kernel_size=3, padding=1),
            nn.GroupNorm(1, hidden),
            nn.ReLU(),
            nn.Conv1d(hidden, n_freq, kernel_size=3, padding=1),
        )

        # 最終 mask head
        self.mask_head = nn.Sequential(
            nn.Conv1d(n_freq, n_freq, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, v_global, spec_mag):
        """
        v_global : [B, 512]   — 視覺全局向量（有 opponent suppression 後的）
        spec_mag : [B, F, T]  — STFT 混音頻譜（magnitude）

        returns:
            mask     [B, F, T]  in (0, 1)
            pred_mag [B, F, T]  = mask × spec_mag
        """
        B, F, T = spec_mag.shape

        # 視覺 FiLM：在頻率維度做 scale + shift（不動 waveform，不造成相位破壞）
        film = self.vis_to_film(v_global)          # [B, F*2]
        gamma = film[:, :F].unsqueeze(2) + 1.0    # [B, F, 1]，初始化在 1 附近
        beta  = film[:, F:].unsqueeze(2)           # [B, F, 1]，初始化在 0 附近

        # 視覺引導的頻譜特徵
        feat = gamma * spec_mag + beta             # [B, F, T]  — FiLM in spec domain
        feat = self.spec_conv(feat)                # [B, F, T]

        # 生成 mask
        mask = self.mask_head(feat)                # [B, F, T] in (0,1)
        pred_mag = mask * spec_mag                 # ratio mask × mixture mag

        return mask, pred_mag


# ============================================================
# 3. PSL Head（Wang et al. ICIP 2022）
#    audio patch cosine similarity vs 視覺全局向量
#    回傳 raw logits 供 BCEWithLogitsLoss 使用
# ============================================================

class PSLHead(nn.Module):
    def __init__(self, audio_dim=512, v_dim=512):
        super().__init__()
        self.audio_proj = nn.Linear(audio_dim, v_dim)

    def forward(self, v_a_global, v_b_global, audio_patches):
        """
        v_a_global   : [B, 512]
        v_b_global   : [B, 512]
        audio_patches: [B, K, 512]
        returns logits [B, K]
        """
        a_proj = self.audio_proj(audio_patches)   # [B, K, 512]
        v_a = F.normalize(v_a_global.unsqueeze(1), dim=-1)  # [B,1,512]
        v_b = F.normalize(v_b_global.unsqueeze(1), dim=-1)  # [B,1,512]
        a_n = F.normalize(a_proj, dim=-1)                   # [B,K,512]
        sim_a = (a_n * v_a).sum(dim=-1)   # [B, K]
        sim_b = (a_n * v_b).sum(dim=-1)   # [B, K]
        return sim_a - sim_b              # raw logits


# ============================================================
# 4. Audio Patch Encoder（用於 PSL）
#    把 STFT 頻譜的時間 frame 作為 patch，投影到 512 維
#    比原本的 waveform patch 更穩定（不會因 waveform 相位而分散）
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
        """
        spec_mag : [B, F, T]
        returns  : [B, T, out_dim]  — 每個時間 frame 是一個 patch
        """
        # spec_mag [B,F,T] → transpose → [B,T,F] → proj → [B,T,512]
        return self.proj(spec_mag.transpose(1, 2))


# ============================================================
# 5. VisualBottleneckInjector
#    在 HTDemucs.crosstransformer 輸出之後注入視覺特徵。
#
#    時間分支 xt [B, C, T_t]：
#      cross-attention(Q=xt_tokens, K/V=visual_seq) + LayerScale 殘差
#
#    頻譜分支 x [B, C, Fr, T_f]：
#      FiLM(visual_global) → gamma/beta 作用在頻率×時間維度 + LayerScale 殘差
#
#    HTDemucs 預設 channels=48, depth=4, growth=2
#    → transformer_channels = 48 * 2^3 = 384
# ============================================================

class VisualBottleneckInjector(nn.Module):
    BOTTLENECK_DIM = 384  # HTDemucs 預設瓶頸 channel 數

    def __init__(self, v_dim=512, num_heads=8):
        super().__init__()
        C = self.BOTTLENECK_DIM

        # 視覺序列投影到音訊瓶頸維度
        self.v_proj = nn.Linear(v_dim, C)

        # 時間分支：cross-attention（Q=audio, K/V=visual）
        self.t_cross_attn = nn.MultiheadAttention(
            embed_dim=C, num_heads=num_heads, batch_first=True
        )
        self.t_norm = nn.LayerNorm(C)

        # 頻譜分支：FiLM
        self.s_film = nn.Linear(v_dim, C * 2)

        # LayerScale：初始接近 0，訓練初期影響極小，不破壞預訓練動態
        self.t_scale = nn.Parameter(torch.full((C,), 1e-4))
        self.s_scale = nn.Parameter(torch.full((C,), 1e-4))

    def forward(self, x, xt, v_seq, v_global):
        """
        x        : [B, C, Fr, T_f]  頻譜分支
        xt       : [B, C, T_t]      時間分支
        v_seq    : [B, T_vis, v_dim] 視覺特徵序列
        v_global : [B, v_dim]        視覺全局向量
        returns  : x_new, xt_new（同 shape）
        """
        B, C, T_t = xt.shape

        # ── 時間分支 ──────────────────────────────────────────
        v_k = self.v_proj(v_seq)                     # [B, T_vis, C]
        xt_q = xt.transpose(1, 2)                    # [B, T_t, C]
        attn_out, _ = self.t_cross_attn(xt_q, v_k, v_k)
        xt_new = self.t_norm(
            xt_q + self.t_scale * attn_out
        ).transpose(1, 2)                            # [B, C, T_t]

        # ── 頻譜分支 ──────────────────────────────────────────
        film  = self.s_film(v_global)                # [B, C*2]
        gamma = film[:, :C].view(B, C, 1, 1)
        beta  = film[:, C:].view(B, C, 1, 1)
        x_new = x + self.s_scale.view(1, C, 1, 1) * (gamma * x + beta)

        return x_new, xt_new


# ============================================================
# 5. 主模型：VisualGuidedHTDemucs（頻譜 masking 版本）
#
#  修改摘要（v3）：
#  ① sources=["guitar_a","guitar_b"]，保留 2-source 輸出
#  ② 移除 waveform-level FiLM（沙沙聲根本來源）
#  ③ 新增 SpectralMaskHead：視覺引導的頻譜 ratio mask
#     - FiLM 移到頻譜域（gamma/beta 作用在 mag spectrogram，不動 phase）
#     - pred_mag = mask × STFT(mix)，iSTFT 重建時用 mix 的 phase
#  ④ HTDemucs 保留作為「可選的波形精修」分支
#     - 若視覺有效：頻譜 mask 分離 + HTDemucs 波形微調（weighted sum）
#     - 若視覺無效：直接用 HTDemucs 輸出（fallback）
#  ⑤ PSL audio patch 改為頻譜 frame patch（更穩定）
# ============================================================

class VisualGuidedHTDemucs(nn.Module):

    # STFT 參數（需與 train 端一致）
    N_FFT = 2048
    HOP   = 512
    N_FREQ = N_FFT // 2 + 1  # 1025

    def __init__(self, visual_weight=0.7):
        """
        visual_weight : 頻譜 mask 分支的混合權重
                        final = visual_weight * spec_out + (1-visual_weight) * htdemucs_out
                        初始設 0.7，讓視覺分支主導，HTDemucs 負責補殘差
        """
        super().__init__()
        self.visual_weight = visual_weight

        # ── 音訊主幹（保留，作為波形精修分支）──────────────────
        self.audio_model = HTDemucs(
            sources=["guitar_a", "guitar_b"],
        )

        # ── 視覺編碼器 ──────────────────────────────────────────
        # data_loader: app=[B,150,13,3] → flatten=39; mot=[B,150,13,4] → flatten=52
        self.app_encoder = AMNetStreamEncoder(input_dim=39)
        self.mot_encoder = AMNetStreamEncoder(input_dim=52)

        # motion→appearance cross-attention（AMNet 精神）
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True,
        )
        # opponent suppression 強度（可學習，初始值 0.3）
        self.alpha = nn.Parameter(torch.tensor(0.3))

        # ── 頻譜 Mask Head（核心，取代 waveform FiLM）──────────
        self.spec_mask_head = SpectralMaskHead(
            v_dim=512, n_freq=self.N_FREQ, hidden=256
        )

        # ── PSL Head ────────────────────────────────────────────
        self.audio_patch_enc = AudioPatchEncoder(n_freq=self.N_FREQ, out_dim=512)
        self.psl_head = PSLHead(audio_dim=512, v_dim=512)

        # ── 瓶頸視覺注入器 ───────────────────────────────────────
        self.bottleneck_injector = VisualBottleneckInjector(v_dim=512, num_heads=8)
        # 注入器狀態（每次 forward 前更新）
        self._vis_seq    = None
        self._vis_global = None
        # monkey-patch：把注入邏輯綁定到 audio_model 上，
        # 在 crosstransformer 之後自動介入
        self._patch_htdemucs_forward()

    # ── HTDemucs monkey-patch ─────────────────────────────────
    def _patch_htdemucs_forward(self):
        """
        把 HTDemucs.forward 替換成注入版本。
        只有 crosstransformer 後那一行不同，其餘與原始碼完全一致。
        """
        from fractions import Fraction
        from einops import rearrange

        outer = self  # 捕捉外部 self

        def patched_forward(htd_self, mix):
            length = mix.shape[-1]
            length_pre_pad = None
            if htd_self.use_train_segment:
                if htd_self.training:
                    htd_self.segment = Fraction(mix.shape[-1], htd_self.samplerate)
                else:
                    training_length = int(htd_self.segment * htd_self.samplerate)
                    if mix.shape[-1] < training_length:
                        length_pre_pad = mix.shape[-1]
                        mix = F.pad(mix, (0, training_length - length_pre_pad))
            z   = htd_self._spec(mix)
            mag = htd_self._magnitude(z).to(mix.device)
            x   = mag
            B, C, Fq, T = x.shape
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            std  = x.std(dim=(1, 2, 3), keepdim=True)
            x    = (x - mean) / (1e-5 + std)
            xt    = mix
            meant = xt.mean(dim=(1, 2), keepdim=True)
            stdt  = xt.std(dim=(1, 2), keepdim=True)
            xt    = (xt - meant) / (1e-5 + stdt)
            saved     = []
            saved_t   = []
            lengths   = []
            lengths_t = []
            for idx, encode in enumerate(htd_self.encoder):
                lengths.append(x.shape[-1])
                inject = None
                if idx < len(htd_self.tencoder):
                    lengths_t.append(xt.shape[-1])
                    tenc = htd_self.tencoder[idx]
                    xt   = tenc(xt)
                    if not tenc.empty:
                        saved_t.append(xt)
                    else:
                        inject = xt
                x = encode(x, inject)
                if idx == 0 and htd_self.freq_emb is not None:
                    frs = torch.arange(x.shape[-2], device=x.device)
                    emb = htd_self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                    x   = x + htd_self.freq_emb_scale * emb
                saved.append(x)
            if htd_self.crosstransformer:
                if htd_self.bottom_channels:
                    b, c, f, t = x.shape
                    x  = rearrange(x, "b c f t-> b c (f t)")
                    x  = htd_self.channel_upsampler(x)
                    x  = rearrange(x, "b c (f t)-> b c f t", f=f)
                    xt = htd_self.channel_upsampler_t(xt)

                x, xt = htd_self.crosstransformer(x, xt)

                # ── 視覺注入（唯一新增的部分）────────────────
                if (outer._vis_seq is not None and
                        outer._vis_global is not None):
                    x, xt = outer.bottleneck_injector(
                        x, xt, outer._vis_seq, outer._vis_global
                    )
                # ─────────────────────────────────────────────

                if htd_self.bottom_channels:
                    x  = rearrange(x, "b c f t-> b c (f t)")
                    x  = htd_self.channel_downsampler(x)
                    x  = rearrange(x, "b c (f t)-> b c f t", f=f)
                    xt = htd_self.channel_downsampler_t(xt)
            for idx, decode in enumerate(htd_self.decoder):
                skip   = saved.pop(-1)
                x, pre = decode(x, skip, lengths.pop(-1))
                offset = htd_self.depth - len(htd_self.tdecoder)
                if idx >= offset:
                    tdec     = htd_self.tdecoder[idx - offset]
                    length_t = lengths_t.pop(-1)
                    if tdec.empty:
                        assert pre.shape[2] == 1, pre.shape
                        pre    = pre[:, :, 0]
                        xt, _  = tdec(pre, None, length_t)
                    else:
                        skip   = saved_t.pop(-1)
                        xt, _  = tdec(xt, skip, length_t)
            assert len(saved) == 0
            assert len(lengths_t) == 0
            assert len(saved_t) == 0
            S = len(htd_self.sources)
            x = x.view(B, S, -1, Fq, T)
            x = x * std[:, None] + mean[:, None]
            x_is_mps = x.device.type == "mps"
            if x_is_mps:
                x = x.cpu()
            zout = htd_self._mask(z, x)
            if htd_self.use_train_segment:
                if htd_self.training:
                    x = htd_self._ispec(zout, length)
                else:
                    x = htd_self._ispec(zout, training_length)
            else:
                x = htd_self._ispec(zout, length)
            if x_is_mps:
                x = x.to("mps")
            if htd_self.use_train_segment:
                if htd_self.training:
                    xt = xt.view(B, S, -1, length)
                else:
                    xt = xt.view(B, S, -1, training_length)
            else:
                xt = xt.view(B, S, -1, length)
            xt = xt * stdt[:, None] + meant[:, None]
            x  = xt + x
            if length_pre_pad:
                x = x[..., :length_pre_pad]
            return x

        import types
        self.audio_model.forward = types.MethodType(patched_forward, self.audio_model)

    def _set_visual(self, v_seq, v_global):
        """HTDemucs forward 執行前呼叫，提供視覺特徵供注入器使用"""
        self._vis_seq    = v_seq      # [B, 150, 512]
        self._vis_global = v_global   # [B, 512]

    def _clear_visual(self):
        """無視覺時清除，注入器自動跳過"""
        self._vis_seq    = None
        self._vis_global = None

    # ── 視覺特徵提取（共用）──────────────────────────────────────
    def encode_visual(self, app, mot):
        """
        app : [B, 150, 13, 3]  → flatten → [B, 150, 39]
        mot : [B, 150, 13, 4]  → flatten → [B, 150, 52]
        returns : [B, 150, 512]
        """
        B = app.shape[0]
        app_feat = self.app_encoder(app.reshape(B, 150, -1))  # [B,150,512]
        mot_feat = self.mot_encoder(mot.reshape(B, 150, -1))  # [B,150,512]
        attn_out, _ = self.cross_attn(mot_feat, app_feat, app_feat)
        return attn_out  # [B, 150, 512]

    # ── Forward ──────────────────────────────────────────────────
    def forward(self, mix_wav, app_A, mot_A, app_B, mot_B, has_visual=True):
        """
        mix_wav   : [B, 1, L]
        app_A/B   : [B, 150, 26, 3]
        mot_A/B   : [B, 150, 26, 4]
        has_visual: bool

        returns:
            pred_wav  [B, 1, L]   — 最終預測波形
            pred_mag  [B, F, T]   — 頻譜 mask 分支的預測 magnitude（用於 spec loss）
            mask      [B, F, T]   — 頻譜 ratio mask，∈(0,1)（用於 IRM BCE loss）
            mix_mag   [B, F, T]   — 混音頻譜 magnitude
            mix_phase [B, F, T]   — 混音相位（iSTFT 用）
            p_hat     [B, T]      — PSL logits（有視覺才有意義）
            has_visual bool
        """
        B, C, L = mix_wav.shape

        # ── STFT 混音 ──────────────────────────────────────────
        mix_mag, mix_phase = stft(mix_wav, n_fft=self.N_FFT, hop=self.HOP)
        # mix_mag: [B, F, T_spec]

        # HTDemucs 需要 stereo input [B, 2, L]
        mix_stereo = mix_wav.repeat(1, 2, 1) if C == 1 else mix_wav

        # ── 視覺引導分支 ───────────────────────────────────────
        if has_visual:
            feat_A = self.encode_visual(app_A, mot_A)  # [B,150,512]
            feat_B = self.encode_visual(app_B, mot_B)  # [B,150,512]

            # opponent suppression
            alpha = torch.clamp(self.alpha, 0.0, 1.0)
            feat_A_final = feat_A - alpha * feat_B     # [B,150,512]
            feat_B_final = feat_B - alpha * feat_A     # [B,150,512]

            # 視覺全局向量（時間維度 mean pool）
            v_a_global = feat_A_final.mean(dim=1)      # [B, 512]
            v_b_global = feat_B_final.mean(dim=1)      # [B, 512]

            # 提供視覺特徵給瓶頸注入器
            self._set_visual(feat_A_final, v_a_global)
        else:
            self._clear_visual()
            v_a_global = None
            v_b_global = None

        # ── HTDemucs 波形分離（瓶頸層視覺注入在內部自動觸發）──
        htd_out = self.audio_model(mix_stereo)
        self._clear_visual()  # forward 結束後清除，避免下次誤用
        # htd_out: [B, 2_sources, 2_stereo, L]
        htd_wav = htd_out[:, 0, :, :].mean(dim=1, keepdim=True)  # [B,1,L]

        if has_visual:

            # 頻譜 ratio mask（FiLM 在頻譜域，不破壞相位）
            mask, pred_mag = self.spec_mask_head(v_a_global, mix_mag)
            # mask: [B,F,T_spec], pred_mag: [B,F,T_spec]

            # iSTFT 重建：pred_mag + mix_phase → 波形（phase 來自原始混音，無相位破壞）
            spec_wav = istft(pred_mag, mix_phase, n_fft=self.N_FFT,
                             hop=self.HOP, length=L)   # [B,1,L]

            # 最終輸出：視覺頻譜分支 × w + HTDemucs 精修 × (1-w)
            pred_wav = self.visual_weight * spec_wav + (1.0 - self.visual_weight) * htd_wav

            # PSL：用頻譜 frame 作為 audio patch（比 waveform patch 更穩定）
            audio_patches = self.audio_patch_enc(mix_mag)   # [B, T_spec, 512]
            p_hat = self.psl_head(v_a_global, v_b_global, audio_patches)  # [B, T_spec]

        else:
            # 無視覺：直接用 HTDemucs 輸出，頻譜分支跳過
            pred_wav = htd_wav
            pred_mag = None
            mask     = None
            p_hat    = None

        return pred_wav, pred_mag, mask, mix_mag, mix_phase, p_hat, has_visual