#with l-spec, l-wav, l-sisdr, l-irm(mask shape)

import os
import torch
import torch.nn.functional as F
import numpy as np
import torchaudio
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

from data_loader_temp import GuitarSeparationDataset
from model_arch import VisualGuidedHTDemucs, stft


# ==========================================
# Metrics（numpy，與 train_base 一致）
# ==========================================

def calculate_metrics(ref, est):
    ref = ref.detach().cpu().numpy()
    est = est.detach().cpu().numpy()
    delta = 1e-7
    dot = np.sum(est * ref, axis=-1, keepdims=True)
    ref_energy = np.sum(ref ** 2, axis=-1, keepdims=True) + delta
    s_target = (dot / ref_energy) * ref
    sdr = 10 * np.log10(np.sum(ref**2, axis=-1) / (np.sum((ref - est)**2, axis=-1) + delta))
    sir = 10 * np.log10(np.sum(s_target**2, axis=-1) / (np.sum((est - s_target)**2, axis=-1) + delta))
    sar = 10 * np.log10(np.sum(s_target**2, axis=-1) / (np.sum((ref - s_target)**2, axis=-1) + delta))
    return np.mean(sdr), np.mean(sir), np.mean(sar)


# ==========================================
# SI-SDR Loss（時域，直接優化 SIR）
# 必須在 autocast 外以 float32 計算，避免數值精度問題
#
# 公式：SI-SDR = 10 * log10(||s_target||² / ||e_noise||²)
#   s_target = <pred, ref> / ||ref||² * ref
#   e_noise  = pred - s_target
# loss = -mean(SI-SDR)，最大化 SI-SDR 等於最小化 -SI-SDR
# ==========================================

def si_sdr_loss(pred, ref, eps=1e-8):
    """
    pred : [B, 1, L]  模型預測
    ref  : [B, 1, L]  ground truth
    returns: scalar loss（負 SI-SDR 的 batch mean）
    """
    # 展平到 [B, L] 方便計算
    pred = pred.squeeze(1).float()   # 強制 float32
    ref  = ref.squeeze(1).float()

    # 零均值化（SI-SDR 的標準做法）
    pred = pred - pred.mean(dim=-1, keepdim=True)
    ref  = ref  - ref.mean(dim=-1, keepdim=True)

    # s_target：ref 在 pred 方向上的投影
    dot        = (pred * ref).sum(dim=-1, keepdim=True)      # [B, 1]
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps  # [B, 1]
    s_target   = (dot / ref_energy) * ref                    # [B, L]

    # e_noise：殘差
    e_noise = pred - s_target                                 # [B, L]

    # SI-SDR per sample
    si_sdr = 10 * torch.log10(
        (s_target * s_target).sum(dim=-1) /
        ((e_noise * e_noise).sum(dim=-1) + eps)
    )  # [B]

    return -si_sdr.mean()   # 最小化負 SI-SDR


# ==========================================
# PSL Ground Truth（Wang et al. 2022）
# 在頻譜 patch 上計算歸屬機率
# ==========================================

def compute_psl_gt(s_a_wav, mix_wav, n_fft=2048, hop=512):
    """
    s_a_wav  : [B, 1, L]
    mix_wav  : [B, 1, L]
    returns p_gt [B, T_spec] in (0,1)
    """
    with torch.no_grad():
        s_b_wav = mix_wav - s_a_wav
        mag_a, _ = stft(s_a_wav, n_fft=n_fft, hop=hop)  # [B, F, T]
        mag_b, _ = stft(s_b_wav, n_fft=n_fft, hop=hop)  # [B, F, T]
        # 每個時間 frame 的能量（沿頻率軸 mean）
        energy_a = mag_a.mean(dim=1)   # [B, T]
        energy_b = mag_b.mean(dim=1)   # [B, T]
        p_gt = torch.sigmoid(energy_a - energy_b)
    return p_gt  # [B, T]


# ==========================================
# Silent Check（RMS-based）
# ==========================================

def is_silent(x, threshold=1e-4):
    """x: [B,1,L] tensor"""
    rms = torch.sqrt(torch.mean(x ** 2))
    return rms < threshold


# ==========================================
# Save Debug
# ==========================================

def save_bad_case(epoch, step, mix, s_a, pred_s_a):
    path = f"{CONFIG['debug_dir']}/epoch{epoch}_step{step}"
    torch.save({"mix": mix.cpu(), "s_a": s_a.cpu(), "pred_s_a": pred_s_a.cpu()},
               path + ".pt")
    torchaudio.save(path + "_mix.wav",    mix[0].cpu(),     CONFIG['sr'])
    torchaudio.save(path + "_predA.wav",  pred_s_a[0].cpu(), CONFIG['sr'])


# ==========================================
# Plot
# ==========================================

def plot_training(history):
    clear_output(wait=True)
    plt.figure(figsize=(12, 10))
    plt.subplot(231); plt.plot(history['loss']);       plt.title("Total Loss")
    plt.subplot(232); plt.plot(history['wav_loss']);   plt.title("Waveform Loss")
    plt.subplot(233); plt.plot(history['spec_loss']);  plt.title("Spectral Loss")
    plt.subplot(234); plt.plot(history['SI-sdr']);     plt.title("SI-SDR")
    plt.subplot(235); plt.plot(history['SI-sir']);     plt.title("SI-SIR")
    plt.subplot(236); plt.plot(history['SI-sar']);     plt.title("SI-SAR")
    plt.tight_layout()
    plt.show()


# ==========================================
# CONFIG
# ==========================================

CONFIG = {
    'data_dir':        '/content/dataset',
    'checkpoint_dir':  '/content/drive/MyDrive/checkpoints_new',
    'debug_dir':       '/content/debug_bad_cases',
    'batch_size':      32,
    'num_epochs':      200,
    'learning_rate':   1e-4,
    'device':          torch.device("cuda"),
    'segment_duration': 5.0,
    'sr':              44100,
    # Loss 權重
    'wav_loss_w':   0.2,   # 波形 L1（輔助）
    'spec_loss_w':  0.3,   # 頻譜 L1（輔助）
    'sisdr_loss_w': 0.5,   # SI-SDR（主力，直接優化 SIR）
}

os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
os.makedirs(CONFIG['debug_dir'],      exist_ok=True)


# ==========================================
# Train
# ==========================================

def train():

    torch.cuda.empty_cache()

    dataset = GuitarSeparationDataset(
        CONFIG['data_dir'],
        segment_duration=CONFIG['segment_duration']
    )

    train_loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=False
    )

    model = VisualGuidedHTDemucs().to(CONFIG['device'])

    # 分層學習率：HTDemucs 主幹用 0.1x，新增的視覺模組用全量 lr
    backbone_params = list(model.audio_model.parameters())
    backbone_ids    = {id(p) for p in backbone_params}
    new_params      = [p for p in model.parameters() if id(p) not in backbone_ids]
    optimizer = AdamW([
        {"params": backbone_params, "lr": CONFIG['learning_rate'] * 0.1},
        {"params": new_params,      "lr": CONFIG['learning_rate']},
    ])
    # CosineAnnealingLR：lr 從 1e-4 平滑降到 1e-6，在第 200 epoch 才到最低點
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'], eta_min=1e-6)
    scaler    = GradScaler()

    history = {
        'loss': [], 'wav_loss': [], 'spec_loss': [],
        'SI-sdr': [], 'SI-sir': [], 'SI-sar': []
    }

    # STFT 參數（需與 model_arch 一致）
    N_FFT = VisualGuidedHTDemucs.N_FFT
    HOP   = VisualGuidedHTDemucs.HOP

    for epoch in range(CONFIG['num_epochs']):

        # PSL weight schedule：前 50 epoch 固定小值暖機，之後線性遞減到 0.1
        # 200 epoch 版：不在前期讓 PSL 主導，讓音訊分離 loss 始終為主
        if epoch < 50:
            psl_weight = 0.5
        else:
            t = (epoch - 50) / 150          # 50→200 epoch 線性遞減
            psl_weight = 0.5 - t * 0.4     # 0.5 → 0.1

        model.train()

        epoch_loss = []
        epoch_wav  = []
        epoch_spec = []
        epoch_sdr  = []
        epoch_sir  = []
        epoch_sar  = []

        pbar = tqdm(train_loader)

        for step, batch in enumerate(pbar):

            mix = batch['mixture'].to(CONFIG['device'])   # [B,1,L]
            s_a = batch['s_a'].to(CONFIG['device'])       # [B,1,L]

            # s_a 靜音跳過（s_b 已在 data_loader 過濾）
            if is_silent(s_a):
                continue

            app_A = batch['app_A'].to(CONFIG['device'])
            mot_A = batch['mot_A'].to(CONFIG['device'])
            app_B = batch['app_B'].to(CONFIG['device'])
            mot_B = batch['mot_B'].to(CONFIG['device'])

            # 樣本權重（RMS-peak-check）
            w = batch['sample_weight'].to(CONFIG['device'])  # [B]

            has_visual = batch.get('has_visual', True)
            if isinstance(has_visual, torch.Tensor):
                has_visual = bool(has_visual.all().item())

            # visual_quality ∈ [0,1]：片段有效偵測幀比例
            # PSL loss 的實際權重 = psl_weight × visual_quality
            # 當偵測率低（大量零值幀）時，PSL 的貢獻自動縮小
            visual_quality = batch.get('visual_quality', torch.ones(1))
            if isinstance(visual_quality, torch.Tensor):
                vq = float(visual_quality.min().item())
            else:
                vq = 1.0
            effective_psl_weight = psl_weight * vq

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):

                # ── 模型前向 ─────────────────────────────────────
                pred_wav, pred_mag, mask, mix_mag, mix_phase, p_hat, _ = model(
                    mix, app_A, mot_A, app_B, mot_B, has_visual=has_visual
                )

                # ── 頻譜 target：s_a 的 magnitude ─────────────────
                s_a_mag, _ = stft(s_a, n_fft=N_FFT, hop=HOP)
                # s_b_mag 用於計算 IRM
                s_b_mag, _ = stft(mix - s_a, n_fft=N_FFT, hop=HOP)

                # ── 波形 L1 loss（低權重，輔助）──────────────────
                l_wav = F.l1_loss(pred_wav, s_a)

                # ── 頻譜 L1 loss（magnitude 域）───────────────────
                if pred_mag is not None:
                    l_spec = F.l1_loss(pred_mag, s_a_mag)
                else:
                    l_spec = l_wav

                # ── IRM mask BCE loss 的 target 在 autocast 內算好 ──
                # （stft 和除法沒問題，只有 BCE 本身不能在 autocast 內）
                if mask is not None and has_visual:
                    eps = 1e-7
                    T_min = min(mask.shape[-1], s_a_mag.shape[-1])
                    irm = (s_a_mag[:, :, :T_min] /
                           (s_a_mag[:, :, :T_min] + s_b_mag[:, :, :T_min] + eps)
                           ).detach()          # [B, F, T_min]
                    mask_for_bce = mask[:, :, :T_min]
                else:
                    irm          = None
                    mask_for_bce = None

                # ── PSL loss（BCEWithLogits，autocast 安全）────────
                if has_visual and p_hat is not None and effective_psl_weight > 0:
                    p_gt = compute_psl_gt(s_a, mix, n_fft=N_FFT, hop=HOP)
                    T_min_psl = min(p_hat.shape[1], p_gt.shape[1])
                    l_psl = F.binary_cross_entropy_with_logits(
                        p_hat[:, :T_min_psl], p_gt[:, :T_min_psl]
                    )
                else:
                    l_psl = torch.tensor(0.0, device=CONFIG['device'])

                # ── Total loss（BCE 部分在 autocast 外計算）────────
                w_mean  = w.mean()
                l_audio = (CONFIG['wav_loss_w'] * l_wav +
                           CONFIG['spec_loss_w'] * l_spec)
                total_loss = (l_audio + effective_psl_weight * l_psl) * w_mean

            # ── IRM BCE loss：在 autocast 區塊外，強制 float32 ────
            if mask_for_bce is not None and irm is not None:
                l_mask = F.binary_cross_entropy(
                    mask_for_bce.float(), irm.float()
                )
                total_loss = total_loss + 2.0 * l_mask * w_mean
            else:
                l_mask = torch.tensor(0.0, device=CONFIG['device'])

            # ── SI-SDR loss：在 autocast 外，float32 保證數值精度 ─
            # 直接優化 SIR，這是讓 SIR 突破負值的關鍵 loss
            l_sisdr = si_sdr_loss(pred_wav, s_a)
            total_loss = total_loss + CONFIG['sisdr_loss_w'] * l_sisdr * w_mean

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred_wav_eval = pred_wav.detach().clamp(-10.0, 10.0)
                sdr, sir, sar = calculate_metrics(s_a, pred_wav_eval)

            # bad case：記錄但不跳過，用 0 代入確保 epoch list 不為空
            if not (np.isfinite(sdr) and np.isfinite(sir) and np.isfinite(sar)):
                print(f"  [bad metrics] step={step} sdr={sdr:.3f}")
                save_bad_case(epoch, step, mix, s_a, pred_wav)
                sdr, sir, sar = 0.0, 0.0, 0.0

            epoch_loss.append(total_loss.item() if np.isfinite(total_loss.item()) else 0.0)
            epoch_wav.append(l_wav.item()   if np.isfinite(l_wav.item())   else 0.0)
            epoch_spec.append(l_spec.item() if np.isfinite(l_spec.item())  else 0.0)
            epoch_sdr.append(float(sdr))
            epoch_sir.append(float(sir))
            epoch_sar.append(float(sar))

            pbar.set_postfix({
                "Loss": f"{total_loss.item():.3f}",
                "Spec": f"{l_spec.item():.3f}",
                "SDR":  f"{sdr:.2f}",
            })

        if not epoch_loss:
            print(f"Epoch {epoch+1}: 沒有有效 step，跳過")
            continue

        avg_loss = np.mean(epoch_loss)
        avg_wav  = np.mean(epoch_wav)
        avg_spec = np.mean(epoch_spec)
        avg_sdr  = np.mean(epoch_sdr)
        avg_sir  = np.mean(epoch_sir)
        avg_sar  = np.mean(epoch_sar)

        history['loss'].append(avg_loss)
        history['wav_loss'].append(avg_wav)
        history['spec_loss'].append(avg_spec)
        history['SI-sdr'].append(avg_sdr)
        history['SI-sir'].append(avg_sir)
        history['SI-sar'].append(avg_sar)

        scheduler.step()

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {avg_loss:.3f}  Wav: {avg_wav:.3f}  Spec: {avg_spec:.3f}")
        print(f"SI-SDR: {avg_sdr:.2f}  SI-SIR: {avg_sir:.2f}  SI-SAR: {avg_sar:.2f}")

        plot_training(history)

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "history": history,
            }, f"{CONFIG['checkpoint_dir']}/model_{epoch+1}.pth")


if __name__ == "__main__":
    train()
