import os
os.environ['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import numpy as np
import torchaudio
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import matplotlib.pyplot as plt
from IPython.display import clear_output

from data_loader_new import GuitarSeparationDataset
from baseline_tasnet import ConvTasNetBaseline


def calculate_metrics(ref, est, mix):
    ref = ref.detach().cpu().numpy().squeeze(1)
    est = est.detach().cpu().numpy().squeeze(1)
    mix = mix.detach().cpu().numpy().squeeze(1)
    delta = 1e-7
    s_b = mix - ref
    dot        = np.sum(est * ref, axis=-1, keepdims=True)
    ref_energy = np.sum(ref ** 2, axis=-1, keepdims=True) + delta
    s_target   = (dot / ref_energy) * ref
    e_noise    = est - s_target
    sdr = 10 * np.log10(np.sum(s_target**2, axis=-1) / (np.sum(e_noise**2, axis=-1) + delta))
    dot_b     = np.sum(e_noise * s_b, axis=-1, keepdims=True)
    sb_energy = np.sum(s_b ** 2, axis=-1, keepdims=True) + delta
    e_interf  = (dot_b / sb_energy) * s_b
    e_artif   = e_noise - e_interf
    sir = 10 * np.log10(np.sum(s_target**2, axis=-1) / (np.sum(e_interf**2, axis=-1) + delta))
    sar = 10 * np.log10(np.sum(s_target**2, axis=-1) / (np.sum(e_artif**2,  axis=-1) + delta))
    return np.mean(sdr), np.mean(sir), np.mean(sar)


def si_sdr_loss(pred, ref, eps=1e-8):
    pred = pred.squeeze(1).float()
    ref  = ref.squeeze(1).float()
    pred = pred - pred.mean(dim=-1, keepdim=True)
    ref  = ref  - ref.mean(dim=-1, keepdim=True)
    dot        = (pred * ref).sum(dim=-1, keepdim=True)
    ref_energy = (ref * ref).sum(dim=-1, keepdim=True) + eps
    s_target   = (dot / ref_energy) * ref
    e_noise    = pred - s_target
    si_sdr = 10 * torch.log10(
        (s_target * s_target).sum(dim=-1) /
        ((e_noise * e_noise).sum(dim=-1) + eps)
    )
    return -si_sdr.mean()


def is_silent(x, threshold=1e-4):
    return torch.sqrt(torch.mean(x ** 2)) < threshold


def save_bad_case(epoch, step, mix, s_a, pred_s_a):
    path = f"{CONFIG['debug_dir']}/epoch{epoch}_step{step}"
    torch.save({"mix": mix.cpu(), "s_a": s_a.cpu(),
                "pred_s_a": pred_s_a.cpu()}, path + ".pt")
    torchaudio.save(path + "_mix.wav",   mix[0].cpu(),      CONFIG['sr'])
    torchaudio.save(path + "_predA.wav", pred_s_a[0].cpu(), CONFIG['sr'])


def plot_training(history):
    clear_output(wait=True)
    plt.figure(figsize=(12, 4))
    plt.subplot(141); plt.plot(history['loss']);    plt.title("Loss")
    plt.subplot(142); plt.plot(history['SI-sdr']);  plt.title("SI-SDR")
    plt.subplot(143); plt.plot(history['SI-sir']);  plt.title("SI-SIR")
    plt.subplot(144); plt.plot(history['SI-sar']);  plt.title("SI-SAR")
    plt.tight_layout()
    plt.show()


CONFIG = {
    'data_dir':         '/content/dataset',
    'checkpoint_dir':   '/content/drive/MyDrive/checkpoints_baseline',
    'debug_dir':        '/content/debug_bad_cases_baseline',
    'batch_size':       16,
    'num_epochs':       200,
    'learning_rate':    1e-4,
    'device':           torch.device("cuda"),
    'segment_duration': 5.0,
    'sr':               44100,
}

os.makedirs(CONFIG['checkpoint_dir'], exist_ok=True)
os.makedirs(CONFIG['debug_dir'],      exist_ok=True)


def train():
    torch.cuda.empty_cache()

    dataset = GuitarSeparationDataset(
        CONFIG['data_dir'],
        segment_duration=CONFIG['segment_duration'],
        split='train',
        seed=42,
    )
    train_loader = DataLoader(
        dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=1,
        pin_memory=False,
    )

    model = ConvTasNetBaseline().to(CONFIG['device'])

    RESUME_CKPT = ""
    start_epoch = 0

    if os.path.exists(RESUME_CKPT):
        ckpt = torch.load(RESUME_CKPT, map_location=CONFIG['device'], weights_only=False)
        model.load_state_dict(ckpt['model'])
        history    = ckpt['history']
        start_epoch = ckpt['epoch'] + 1
        print(f"✅ 從 epoch {start_epoch} 繼續訓練")
    else:
        print("從頭開始訓練")

    optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'],
                                  eta_min=1e-8)
    scaler    = GradScaler()

    for _ in range(start_epoch):
        scheduler.step()

    if start_epoch == 0:
        history = {'loss': [], 'SI-sdr': [], 'SI-sir': [], 'SI-sar': []}

    for epoch in range(start_epoch, CONFIG['num_epochs']):
        model.train()
        epoch_loss, epoch_sdr, epoch_sir, epoch_sar = [], [], [], []

        pbar = tqdm(train_loader)
        for step, batch in enumerate(pbar):

            mix = batch['mixture'].to(CONFIG['device'])
            s_a = batch['s_a'].to(CONFIG['device'])
            w   = batch['sample_weight'].to(CONFIG['device'])

            if is_silent(s_a):
                continue

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                pred_wav = model(mix)
                loss = si_sdr_loss(pred_wav, s_a) * w.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            with torch.no_grad():
                pred_eval = pred_wav.detach().clamp(-10.0, 10.0)
                sdr, sir, sar = calculate_metrics(s_a, pred_eval, mix)

            if not (np.isfinite(sdr) and np.isfinite(sir) and np.isfinite(sar)):
                print(f"  [bad metrics] step={step}")
                save_bad_case(epoch, step, mix, s_a, pred_wav)
                sdr, sir, sar = 0.0, 0.0, 0.0

            epoch_loss.append(loss.item() if np.isfinite(loss.item()) else 0.0)
            epoch_sdr.append(float(sdr))
            epoch_sir.append(float(sir))
            epoch_sar.append(float(sar))

            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "SDR":  f"{sdr:.2f}",
                "SIR":  f"{sir:.2f}",
                "SAR":  f"{sar:.2f}",
            })

        if not epoch_loss:
            print(f"Epoch {epoch+1}: 沒有有效 step，跳過")
            continue

        avg_loss = np.mean(epoch_loss)
        avg_sdr  = np.mean(epoch_sdr)
        avg_sir  = np.mean(epoch_sir)
        avg_sar  = np.mean(epoch_sar)

        history['loss'].append(avg_loss)
        history['SI-sdr'].append(avg_sdr)
        history['SI-sir'].append(avg_sir)
        history['SI-sar'].append(avg_sar)

        scheduler.step()

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {avg_loss:.3f}")
        print(f"SI-SDR: {avg_sdr:.2f}  SI-SIR: {avg_sir:.2f}  SI-SAR: {avg_sar:.2f}")

        plot_training(history)

        if (epoch + 1) % 5 == 0:
            torch.save({
                "epoch":   epoch,
                "model":   model.state_dict(),
                "history": history,
            }, f"{CONFIG['checkpoint_dir']}/baseline_{epoch+1}.pth")

        torch.save({
            "epoch":   epoch,
            "model":   model.state_dict(),
            "history": history,
        }, f"{CONFIG['checkpoint_dir']}/baseline_latest.pth")


if __name__ == "__main__":
    train()