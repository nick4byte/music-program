import torch
import torch.nn as nn
import torch.nn.functional as F
from demucs.htdemucs import HTDemucs


def stft(wav, n_fft=2048, hop=512):
    """wav [B,1,L] → mag [B,F,T], phase [B,F,T]"""
    x   = wav.squeeze(1)
    win = torch.hann_window(n_fft, device=wav.device)
    spec = torch.stft(x, n_fft=n_fft, hop_length=hop,
                      window=win, return_complex=True, normalized=False)
    return spec.abs(), spec.angle()


class PureHTDemucs(nn.Module):


    N_FFT  = 2048   # 與 model_arch_new.py 一致
    HOP    = 512
    N_FREQ = N_FFT // 2 + 1

    def __init__(self):
        super().__init__()
        self.audio_model = HTDemucs(
            sources=["guitar_a", "guitar_b"]
        )

    def forward(self, mix_wav):
        """
        mix_wav : [B, 1, L]
        returns:
            pred_wav  [B, 1, L]   
            pred_mag  [B, F, T]   
            mix_mag   [B, F, T]   
        """
        B, C, L = mix_wav.shape

        mix_stereo = mix_wav.repeat(1, 2, 1) if C == 1 else mix_wav

        out     = self.audio_model(mix_stereo)   
        pred_a  = out[:, 0].mean(dim=1, keepdim=True)  


        mix_mag, _ = stft(mix_wav, n_fft=self.N_FFT, hop=self.HOP)
        pred_mag, _ = stft(pred_a, n_fft=self.N_FFT, hop=self.HOP)

        return pred_a, pred_mag, mix_mag
