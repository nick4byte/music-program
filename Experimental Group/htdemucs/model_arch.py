import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from demucs.htdemucs import HTDemucs


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
    mag   = spec.abs()           
    phase = spec.angle()          
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
    return wav.unsqueeze(1)      


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
        return self.net(x)  


class SpectralMaskHead(nn.Module):
    def __init__(self, v_dim=512, n_freq=1025, hidden=256):
        super().__init__()
        self.n_freq = n_freq

        self.vis_to_film = nn.Sequential(
            nn.Linear(v_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_freq * 2),  
        )

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

    def forward(self, v_global, spec_mag):
        B, F, T = spec_mag.shape

        film = self.vis_to_film(v_global)        
        gamma = film[:, :F].unsqueeze(2) + 1.0    
        beta  = film[:, F:].unsqueeze(2)           


        feat = gamma * spec_mag + beta             
        feat = self.spec_conv(feat)                


        mask = self.mask_head(feat)               
        pred_mag = mask * spec_mag                 

        return mask, pred_mag


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
        a_proj = self.audio_proj(audio_patches)   
        v_a = F.normalize(v_a_global.unsqueeze(1), dim=-1)  
        v_b = F.normalize(v_b_global.unsqueeze(1), dim=-1)  
        a_n = F.normalize(a_proj, dim=-1)                   
        sim_a = (a_n * v_a).sum(dim=-1)   
        sim_b = (a_n * v_b).sum(dim=-1)   
        return sim_a - sim_b              




class AudioPatchEncoder(nn.Module):
    def __init__(self, n_freq=1025, out_dim=512):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(n_freq, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, spec_mag):
        return self.proj(spec_mag.transpose(1, 2))




class VisualBottleneckInjector(nn.Module):
    BOTTLENECK_DIM = 384  

    def __init__(self, v_dim=512, num_heads=8):
        super().__init__()
        C = self.BOTTLENECK_DIM

        
        self.v_proj = nn.Linear(v_dim, C)

       
        self.t_cross_attn = nn.MultiheadAttention(
            embed_dim=C, num_heads=num_heads, batch_first=True
        )
        self.t_norm = nn.LayerNorm(C)

        
        self.s_film = nn.Linear(v_dim, C * 2)

        
        self.t_scale = nn.Parameter(torch.full((C,), 1e-4))
        self.s_scale = nn.Parameter(torch.full((C,), 1e-4))

    def forward(self, x, xt, v_seq, v_global):
        B, C, T_t = xt.shape

        v_k = self.v_proj(v_seq)                     
        xt_q = xt.transpose(1, 2)                    
        attn_out, _ = self.t_cross_attn(xt_q, v_k, v_k)
        xt_new = self.t_norm(
            xt_q + self.t_scale * attn_out
        ).transpose(1, 2)                            

        film  = self.s_film(v_global)                
        gamma = film[:, :C].view(B, C, 1, 1)
        beta  = film[:, C:].view(B, C, 1, 1)
        x_new = x + self.s_scale.view(1, C, 1, 1) * (gamma * x + beta)

        return x_new, xt_new



class VisualGuidedHTDemucs(nn.Module):

    N_FFT = 2048
    HOP   = 512
    N_FREQ = N_FFT // 2 + 1  # 1025

    def __init__(self, visual_weight=0.7):
        super().__init__()
        self.visual_weight = visual_weight

        self.audio_model = HTDemucs(
            sources=["guitar_a", "guitar_b"],
        )

        self.app_encoder = AMNetStreamEncoder(input_dim=39)
        self.mot_encoder = AMNetStreamEncoder(input_dim=52)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=512, num_heads=8, batch_first=True,
        )
        self.alpha = nn.Parameter(torch.tensor(0.3))

        self.spec_mask_head = SpectralMaskHead(
            v_dim=512, n_freq=self.N_FREQ, hidden=256
        )
        self.audio_patch_enc = AudioPatchEncoder(n_freq=self.N_FREQ, out_dim=512)
        self.psl_head = PSLHead(audio_dim=512, v_dim=512)

        self.bottleneck_injector = VisualBottleneckInjector(v_dim=512, num_heads=8)

        self._vis_seq    = None
        self._vis_global = None
        self._patch_htdemucs_forward()

    def _patch_htdemucs_forward(self):
        from fractions import Fraction
        from einops import rearrange

        outer = self  

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

                if (outer._vis_seq is not None and
                        outer._vis_global is not None):
                    x, xt = outer.bottleneck_injector(
                        x, xt, outer._vis_seq, outer._vis_global
                    )

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
        self._vis_seq    = v_seq      # [B, 150, 512]
        self._vis_global = v_global   # [B, 512]

    def _clear_visual(self):
        self._vis_seq    = None
        self._vis_global = None

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
        
    def forward(self, mix_wav, app_A, mot_A, app_B, mot_B, has_visual=True):
        B, C, L = mix_wav.shape

        mix_mag, mix_phase = stft(mix_wav, n_fft=self.N_FFT, hop=self.HOP)
        # mix_mag: [B, F, T_spec]

        # HTDemucs 需要 stereo input [B, 2, L]
        mix_stereo = mix_wav.repeat(1, 2, 1) if C == 1 else mix_wav

        if has_visual:
            feat_A = self.encode_visual(app_A, mot_A)  # [B,150,512]
            feat_B = self.encode_visual(app_B, mot_B)  # [B,150,512]

            alpha = torch.clamp(self.alpha, 0.0, 1.0)
            feat_A_final = feat_A - alpha * feat_B     # [B,150,512]
            feat_B_final = feat_B - alpha * feat_A     # [B,150,512]


            v_a_global = feat_A_final.mean(dim=1)      # [B, 512]
            v_b_global = feat_B_final.mean(dim=1)      # [B, 512]

            self._set_visual(feat_A_final, v_a_global)
        else:
            self._clear_visual()
            v_a_global = None
            v_b_global = None

        htd_out = self.audio_model(mix_stereo)
        self._clear_visual() 
        # htd_out: [B, 2_sources, 2_stereo, L]
        htd_wav = htd_out[:, 0, :, :].mean(dim=1, keepdim=True)  # [B,1,L]

        if has_visual:

            mask, pred_mag = self.spec_mask_head(v_a_global, mix_mag)
            spec_wav = istft(pred_mag, mix_phase, n_fft=self.N_FFT,
                             hop=self.HOP, length=L)   # [B,1,L]

            pred_wav = self.visual_weight * spec_wav + (1.0 - self.visual_weight) * htd_wav

            audio_patches = self.audio_patch_enc(mix_mag)   # [B, T_spec, 512]
            p_hat = self.psl_head(v_a_global, v_b_global, audio_patches)  # [B, T_spec]

        else:
            pred_wav = htd_wav
            pred_mag = None
            mask     = None
            p_hat    = None

        return pred_wav, pred_mag, mask, mix_mag, mix_phase, p_hat, has_visual
