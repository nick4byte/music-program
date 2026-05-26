#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import librosa
from torch.utils.data import Dataset
from pedalboard import Pedalboard, Compressor, Reverb, HighShelfFilter, Bitcrush


class GuitarSeparationDataset(Dataset):
    def __init__(self, data_dir, segment_duration=5.0, sample_rate=44100,
                 min_energy_threshold=1e-4, split='train', test_ratio=0.2, seed=42):
        self.data_dir = data_dir
        self.audio_root = os.path.join(data_dir, "audio")
        self.visual_root = os.path.join(data_dir, "visual")
        self.segment_duration = segment_duration
        self.sr = sample_rate
        self.target_frames = 150
        self.min_energy = min_energy_threshold

        all_samples = [
            f.replace("_sync.json", "")
            for f in os.listdir(self.visual_root)
            if f.endswith("_sync.json")
        ]

        rng     = np.random.RandomState(seed)
        indices = rng.permutation(len(all_samples))
        n_test  = int(len(all_samples) * test_ratio)

        if split == 'train':
            self.samples = [all_samples[i] for i in indices[n_test:]]
        else:
            self.samples = [all_samples[i] for i in indices[:n_test]]

        print(f"✅ 載入 {split} set: {len(self.samples)} 個樣本")

        # ── 1. 樣本權重預載（小檔案，全部載入）─────────────────────
        self._weight_cache = {}
        for s_id in self.samples:
            weight_file = os.path.join(self.visual_root, f"{s_id}_weight.npy")
            self._weight_cache[s_id] = float(np.load(weight_file)) if os.path.exists(weight_file) else 1.0

        # ── 2. 音訊預載到記憶體 ────────────────────────────────────
        # librosa.load 是最大的瓶頸（每次 0.5~2 秒）。
        # 31 個樣本 × 3~5 分鐘 × 44100Hz × 4B ≈ 300~600MB，完全可行。
        print("預載音訊中...")
        self._audio_cache = {}
        for s_id in self.samples:
            path = self._find_file(self.audio_root, s_id, ".wav")
            self._audio_cache[s_id] = librosa.load(path, sr=self.sr)[0]
        print(f"  音訊預載完成：{len(self._audio_cache)} 個")

        # ── 3. valid mask 預載（很小，全部載入）────────────────────
        self._valid_cache = {}
        for s_id in self.samples:
            valid_path = os.path.join(self.visual_root, f"{s_id}_valid.npy")
            if os.path.exists(valid_path):
                self._valid_cache[s_id] = np.load(valid_path)
            else:
                n_frames = int(len(self._audio_cache[s_id]) / self.sr * 30)
                self._valid_cache[s_id] = np.ones(n_frames, dtype=bool)

        # ── 4. 視覺 npy 全部預載到記憶體 ──────────────────────────
        # _A_Left.npy / _A_Right.npy：shape 通常是 [T_frames, 13, 7]。
        # mmap 雖然省記憶體，但每次 __getitem__ 開關 mmap 仍有 overhead。
        # 直接全載後 __getitem__ 只做 numpy slicing，速度最快。
        print("預載視覺特徵中...")
        self._visual_cache = {}   # { s_id: np.ndarray [T, joints, dim] }
        for s_id in self.samples:
            v_l = np.load(self._find_file(self.visual_root, s_id, "_A_Left.npy"))
            v_r = np.load(self._find_file(self.visual_root, s_id, "_A_Right.npy"))
            self._visual_cache[s_id] = np.concatenate([v_l, v_r], axis=1)
        print(f"  視覺特徵預載完成：{len(self._visual_cache)} 個")

        print("✅ 所有資料預載完成")

    def __len__(self):
        return len(self.samples) * 10

    # ── 靜音判斷 ────────────────────────────────────────────────
    def _is_silent(self, wav_np):
        return np.sqrt(np.mean(wav_np ** 2)) < self.min_energy

    # ── 樣本權重（從 cache 取，不讀檔）─────────────────────────
    def _load_sample_weight(self, s_id):
        return self._weight_cache.get(s_id, 1.0)

    # ── 隨機後製 ────────────────────────────────────────────────
    def apply_random_postprocessing(self, waveform_np):
        board = Pedalboard([
            Compressor(
                threshold_db=np.random.uniform(-30, -12),
                ratio=np.random.uniform(2, 8)
            ),
            Reverb(wet_level=np.random.uniform(0.0, 0.25)),
            HighShelfFilter(gain_db=np.random.uniform(-6, 6)),
            Bitcrush(bit_depth=float(np.random.choice([16, 24])))
        ])
        return board(waveform_np.astype(np.float32), self.sr)

    def _find_file(self, root, s_id, suffix):
        """初始化時使用，__getitem__ 不再呼叫此函式"""
        for f in os.listdir(root):
            if f.startswith(s_id) and f.endswith(suffix):
                return os.path.join(root, f)
        raise FileNotFoundError(f"找不到 {s_id} 的 {suffix} 檔案")

    def process_v_optimized(self, m_raw, start_f):
        actual_req_frames = int(self.segment_duration * 30)
        end_f = min(start_f + actual_req_frames, m_raw.shape[0])
        m_crop = m_raw[start_f: end_f]

        if len(m_crop) < actual_req_frames:
            padding = np.zeros((actual_req_frames - len(m_crop),
                                m_raw.shape[1], m_raw.shape[2]))
            m_crop = np.concatenate([m_crop, padding], axis=0)

        v_tensor = torch.from_numpy(m_crop).float().reshape(m_crop.shape[0], -1).t().unsqueeze(0)
        v_res = F.interpolate(v_tensor, size=self.target_frames,
                              mode='linear', align_corners=False)
        return v_res.squeeze(0).t().reshape(self.target_frames, m_raw.shape[1], -1)

    def __getitem__(self, index):
        try:
            s_id = self.samples[index % len(self.samples)]
            target_samples = int(self.segment_duration * self.sr)

            # ── Guitar A ─────────────────────────────────────────
            audio_full   = self._audio_cache[s_id]       # 記憶體，無 I/O
            valid_mask_A = self._valid_cache[s_id]        # 記憶體，無 I/O
            m_raw_A      = self._visual_cache[s_id]       # 記憶體，無 I/O
            max_start = max(0, len(audio_full) - target_samples)

            # 找出有足夠有效幀覆蓋的起始位置（最多嘗試 10 次）
            start_s = random.randint(0, max_start)
            for _ in range(10):
                start_f_try = int((start_s / self.sr) * 30)
                end_f_try   = start_f_try + int(self.segment_duration * 30)
                seg_valid   = valid_mask_A[start_f_try:end_f_try]
                # 至少 50% 的幀有有效偵測才接受
                if len(seg_valid) == 0 or seg_valid.mean() >= 0.5:
                    break
                start_s = random.randint(0, max_start)
            start_f = int((start_s / self.sr) * 30)

            s_a = audio_full[start_s: start_s + target_samples]
            s_a = np.pad(s_a, (0, max(0, target_samples - len(s_a))))[:target_samples]

            if self._is_silent(s_a):
                return self.__getitem__(random.randint(0, len(self.samples) - 1))

            m_a = self.process_v_optimized(m_raw_A, start_f)

            # 計算 Guitar A 這段片段的視覺有效率
            seg_valid_a  = valid_mask_A[start_f: start_f + int(self.segment_duration * 30)]
            visual_quality_a = float(seg_valid_a.mean()) if len(seg_valid_a) > 0 else 0.0

            # ── Guitar B（加入靜音檢查，最多重試 5 次）────────────
            s_b = None
            m_b = None
            visual_quality_b = 0.0
            for _ in range(5):
                idx_B = random.randint(0, len(self.samples) - 1)
                while idx_B == index % len(self.samples):
                    idx_B = random.randint(0, len(self.samples) - 1)
                s_id_B = self.samples[idx_B]

                audio_full_B = self._audio_cache[s_id_B]  # 記憶體，無 I/O
                start_s_B = random.randint(0, max(0, len(audio_full_B) - target_samples))
                start_f_B = int((start_s_B / self.sr) * 30)

                candidate_b = audio_full_B[start_s_B: start_s_B + target_samples]
                candidate_b = np.pad(
                    candidate_b, (0, max(0, target_samples - len(candidate_b)))
                )[:target_samples]

                if self._is_silent(candidate_b):
                    continue

                # Guitar B 的有效幀率（從 cache 取）
                valid_mask_B = self._valid_cache[s_id_B]  # 記憶體，無 I/O
                ef_b = start_f_B
                et_b = ef_b + int(self.segment_duration * 30)
                seg_b = valid_mask_B[ef_b:et_b]
                vq_b  = float(seg_b.mean()) if len(seg_b) > 0 else 0.0

                m_raw_B = self._visual_cache[s_id_B]       # 記憶體，無 I/O

                s_b = candidate_b
                m_b = self.process_v_optimized(m_raw_B, start_f_B)
                visual_quality_b = vq_b
                break

            if s_b is None:
                return self.__getitem__(random.randint(0, len(self.samples) - 1))

            # ── PSL velocity ────────────────────────────────────────
            PICKING_HAND_IDX = 11

            # ── 混合 + 後製 ─────────────────────────────────────────
            mixture = s_a + s_b
            mixture = self.apply_random_postprocessing(mixture)

            # ── 樣本權重 ────────────────────────────────────────────
            weight_a = self._load_sample_weight(s_id)
            weight_b = self._load_sample_weight(s_id_B)
            sample_weight = (weight_a + weight_b) / 2.0

            # ── 視覺有效率：A 和 B 取最小值 ────────────────────────
            # has_visual=True  → 視覺特徵足夠可靠，PSL loss 以完整權重計算
            # has_visual=False → 大量零值幀，PSL loss 應跳過或降權
            visual_quality = min(visual_quality_a, visual_quality_b)
            has_visual = bool(visual_quality >= 0.5)

            return {
                "mixture":        torch.from_numpy(mixture).float().unsqueeze(0),
                "s_a":            torch.from_numpy(s_a).float().unsqueeze(0),
                "app_A":          m_a[:, :, :3].float(),
                "mot_A":          m_a[:, :, 3:7].float(),
                "vel_A":          m_a[:, PICKING_HAND_IDX, 6].float(),
                "app_B":          m_b[:, :, :3].float(),
                "mot_B":          m_b[:, :, 3:7].float(),
                "vel_B":          m_b[:, PICKING_HAND_IDX, 6].float(),
                "sample_weight":  torch.tensor(sample_weight, dtype=torch.float32),
                "has_visual":     torch.tensor(has_visual, dtype=torch.bool),
                "visual_quality": torch.tensor(visual_quality, dtype=torch.float32),
            }

        except Exception as e:
            print(f"Error loading index {index}: {e}")
            return self.__getitem__(random.randint(0, len(self.samples) - 1))
