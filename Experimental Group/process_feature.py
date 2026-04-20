import cv2
import mediapipe as mp
import numpy as np
import librosa
import soundfile as sf
import json
import os
from collections import deque
import subprocess
from pathlib import Path

mp_hands    = mp.solutions.hands
mp_drawing  = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


class HandSmoothingFilter:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.buffer = deque(maxlen=window_size)

    def update(self, coords):
        self.buffer.append(coords)
        if len(self.buffer) < 3:
            return coords
        return np.median(np.array(self.buffer), axis=0)

    def reset(self):
        self.buffer.clear()


def extract_and_struct_data(results, prev_coords_dict, filters, frame_idx=0):
    LEFT_INDICES  = [5, 8, 9, 12, 13, 16, 17, 20]   # 8 個關節（按弦手）
    RIGHT_INDICES = [0, 2, 4, 5, 9]                  # 5 個關節（撥弦手）

    frame_struct = {
        "A_Left":  np.zeros((8, 7)),
        "A_Right": np.zeros((5, 7)),
        "valid":   False,          # 此幀是否有有效偵測
    }

    if not results.multi_hand_landmarks:
        for key in ["A_Left", "A_Right"]:
            filters[key].reset()
        return frame_struct, prev_coords_dict

    detected_hands = []
    for res_lms, res_label in zip(results.multi_hand_landmarks, results.multi_handedness):
        wrist = res_lms.landmark[0]
        detected_hands.append({
            'landmarks': res_lms,
            'wrist_x':   wrist.x,
        })

    # 按 X 座標排序：最左邊的是畫面左側（撥弦手/Right），最右邊是按弦手/Left
    detected_hands = sorted(detected_hands, key=lambda h: h['wrist_x'])

    for i, hand_info in enumerate(detected_hands):
        res_lms = hand_info['landmarks']
        wrist_x = hand_info['wrist_x']

        if wrist_x < 0.4:
            actual_hand = "Right"
            indices     = RIGHT_INDICES
        else:
            actual_hand = "Left"
            indices     = LEFT_INDICES

        key = f"A_{actual_hand}"

        # 提取相對腕關節的座標（xyz）
        wrist = res_lms.landmark[0]
        raw_coords = np.array([
            [res_lms.landmark[idx].x - wrist.x,
             res_lms.landmark[idx].y - wrist.y,
             res_lms.landmark[idx].z - wrist.z]
            for idx in indices
        ])

        # 平滑
        try:
            smooth_coords = filters[key].update(raw_coords)
        except Exception:
            smooth_coords = raw_coords

        # 運動差分（dx, dy, dz）
        prev_coords = prev_coords_dict.get(key, np.zeros_like(smooth_coords))
        diff = smooth_coords - prev_coords

        velocity_raw = np.linalg.norm(diff, axis=-1, keepdims=True)  # [points, 1]

        motion_data = np.concatenate([diff, velocity_raw], axis=-1)  # [points, 4]
        frame_struct[key] = np.concatenate([smooth_coords, motion_data], axis=1)  # [points, 7]
        prev_coords_dict[key] = smooth_coords

    # 只要至少偵測到一隻手就標記為有效
    frame_struct["valid"] = True
    return frame_struct, prev_coords_dict


def draw_tracking_frame(frame, results, frame_idx):
    annotated = frame.copy()

    if not results.multi_hand_landmarks:
        cv2.putText(annotated, f"F{frame_idx}: No hands",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return annotated

    for res_lms, res_label in zip(results.multi_hand_landmarks, results.multi_handedness):
        wrist_x    = res_lms.landmark[0].x
        confidence = res_label.classification[0].score

        # 與 extract_and_struct_data 一致的判斷邏輯
        if wrist_x < 0.4:
            label      = "RIGHT(picking)"
            color      = (0, 0, 255)    # 紅 = 撥弦手
        else:
            label      = "LEFT(fretting)"
            color      = (255, 100, 0)  # 藍 = 按弦手

        # 繪製骨架（用自訂顏色覆蓋預設樣式）
        h, w = annotated.shape[:2]
        for connection in mp_hands.HAND_CONNECTIONS:
            s_idx, e_idx = connection
            s = res_lms.landmark[s_idx]
            e = res_lms.landmark[e_idx]
            sx, sy = int(s.x * w), int(s.y * h)
            ex, ey = int(e.x * w), int(e.y * h)
            cv2.line(annotated, (sx, sy), (ex, ey), color, 2)

        # 繪製關節點
        for lm in res_lms.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(annotated, (cx, cy), 4, color, -1)

        # 腕關節旁標注 label
        wrist_px = int(res_lms.landmark[0].x * w)
        wrist_py = int(res_lms.landmark[0].y * h)
        cv2.putText(annotated, f"{label} {confidence:.2f}",
                    (wrist_px + 8, wrist_py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # 右上角幀號
    cv2.putText(annotated, f"F{frame_idx}",
                (annotated.shape[1] - 80, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    return annotated


def video_audio_process(video_path, audio_path, output_prefix=None,
                        write_tracking_video=True, tracking_fps=None):
    if output_prefix is None:
        output_prefix = Path(video_path).stem
        print(f"📝 輸出前綴: {output_prefix}")

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"影片不存在: {video_path}")
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"音訊不存在: {audio_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"無法開啟影片: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    target_sr    = 44100

    print(f"\n🎬 影片資訊: {frame_width}×{frame_height} @ {fps:.2f}fps, {total_frames}幀")

    audio_full, _ = librosa.load(audio_path, sr=target_sr, mono=True)
    print(f"🎵 音訊長度: {len(audio_full)/target_sr:.2f} 秒")

    filters = {
        "A_Left":  HandSmoothingFilter(window_size=5),
        "A_Right": HandSmoothingFilter(window_size=5),
    }
    data_storage    = {"A_Left": [], "A_Right": []}
    valid_mask_list = []
    prev_coords_dict = {}
    processed_frames = 0

    # ── VideoWriter 初始化（追蹤視覺化）───────────────────────────
    video_writer = None
    if write_tracking_video:
        out_fps    = tracking_fps if tracking_fps else fps
        out_path   = f"{output_prefix}_tracking.mp4"
        fourcc     = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            out_path, fourcc, out_fps, (frame_width, frame_height)
        )
        print(f"🎥 追蹤影片輸出: {out_path}")

    print("\n🎸 開始逐幀處理...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_data, prev_coords_dict = extract_and_struct_data(
            results, prev_coords_dict, filters, frame_idx=processed_frames
        )

        for key in data_storage:
            data_storage[key].append(frame_data[key])
        valid_mask_list.append(frame_data["valid"])

        # ── 寫入追蹤幀 ─────────────────────────────────────────
        if video_writer is not None:
            annotated = draw_tracking_frame(frame, results, processed_frames)
            video_writer.write(annotated)

        processed_frames += 1
        if processed_frames % 100 == 0:
            print(f"   進度: {processed_frames}/{total_frames}")

    cap.release()
    if video_writer is not None:
        video_writer.release()
        print(f"✅ 追蹤影片已儲存: {out_path}")

    valid_mask = np.array(valid_mask_list, dtype=bool)
    valid_ratio = valid_mask.mean() * 100
    print(f"\n✅ 處理完成: {processed_frames}幀, 有效偵測率: {valid_ratio:.1f}%")

    # ── 全局速度歸一化（修正②：在 pre-clip 移除後，這裡才有意義）──
    print("\n📏 全域速度歸一化...")
    for key in ["A_Right", "A_Left"]:
        if data_storage[key]:
            arr   = np.array(data_storage[key])          # [F, points, 7]
            v_all = arr[:, :, 6]                          # 速度欄位
            max_v = np.max(v_all)
            if max_v > 0:
                arr[:, :, 6] = np.clip(v_all / max_v, 0.0, 1.0)
                print(f"   {key}: max_v={max_v:.4f} → 歸一化後 [0,1]")
            data_storage[key] = arr.tolist()

    # ── 資料檢查 ─────────────────────────────────────────────────
    print("\n🔍 資料統計:")
    for key in data_storage:
        arr = np.array(data_storage[key])
        non_zero = np.sum(np.any(arr != 0, axis=(1, 2)))
        print(f"   {key}: shape={arr.shape}, 有資料幀={non_zero}/{processed_frames} "
              f"({non_zero/processed_frames*100:.1f}%)")
        if non_zero > 0:
            first = np.where(np.any(arr != 0, axis=(1, 2)))[0][0]
            s = arr[first]
            print(f"      x∈[{s[:,0].min():.3f},{s[:,0].max():.3f}] "
                  f"V∈[{s[:,6].min():.3f},{s[:,6].max():.3f}]")

    # ── 裁切音訊 ─────────────────────────────────────────────────
    actual_len  = int(processed_frames * (target_sr / fps))
    audio_final = audio_full[:actual_len]

    # ── LUFS 正規化 + RMS 動態範圍計算 ───────────────────────────
    print("\n📊 音量正規化與 RMS 檢查...")
    temp_in  = f"{output_prefix}_temp_in.wav"
    temp_out = f"{output_prefix}_temp_out.wav"
    sf.write(temp_in, audio_final, target_sr)
    subprocess.run(
        ["ffmpeg", "-y", "-i", temp_in,
         "-af", "loudnorm=I=-16:TP=-1:LRA=11",
         "-ar", str(target_sr), "-ac", "1", temp_out],
        capture_output=True
    )
    audio_final, _ = sf.read(temp_out)
    if audio_final.ndim > 1:
        audio_final = audio_final.mean(axis=1)
    os.remove(temp_in)

    peak             = np.max(np.abs(audio_final))
    rms              = np.sqrt(np.mean(audio_final ** 2)) + 1e-8
    dynamic_range_db = 20 * np.log10(peak / rms)

    if dynamic_range_db < 9:
        sample_weight = 0.3
    elif dynamic_range_db < 12:
        sample_weight = 0.7
    else:
        sample_weight = 1.0
    print(f"   Peak-to-RMS: {dynamic_range_db:.2f} dB → weight={sample_weight}")

    # ── 儲存 ─────────────────────────────────────────────────────
    sf.write(f"{output_prefix}_A.wav", audio_final, target_sr)
    print(f"\n💾 音訊: {output_prefix}_A.wav")

    for key in data_storage:
        arr = np.array(data_storage[key])
        np.save(f"{output_prefix}_{key}.npy", arr)
        print(f"   {key}: {arr.shape} → {output_prefix}_{key}.npy")

    # 儲存有效幀 mask，供 data_loader 跳過零值幀用
    np.save(f"{output_prefix}_valid.npy", valid_mask)
    print(f"   valid_mask: {valid_mask.shape}, 有效率={valid_ratio:.1f}%")

    # 同時儲存樣本權重為獨立 npy（data_loader 直接 load）
    np.save(f"{output_prefix}_weight.npy", np.array(sample_weight))
    print(f"   weight: {sample_weight}")

    sync_data = {
        "metadata": {
            "video_source":       video_path,
            "audio_source":       audio_path,
            "video_resolution":   f"{frame_width}x{frame_height}",
            "audio_total_samples": len(audio_final),
            "sample_rate":        target_sr,
            "fps":                fps,
            "total_frames":       processed_frames,
            "valid_frames":       int(valid_mask.sum()),
            "valid_ratio":        round(float(valid_ratio), 2),
            "feature_dim":        7,
            "feature_description": "x,y,z (rel. to wrist), dx,dy,dz (motion diff), V (velocity, global-normalized)",
            "left_hand_points":   8,
            "right_hand_points":  5,
            "picking_hand":       "A_Right",
            "fretting_hand":      "A_Left",
            "dynamic_range_db":   round(float(dynamic_range_db), 2),
            "sample_weight":      sample_weight,
        },
        "sync_index": [int(np.round(i * (target_sr / fps))) for i in range(processed_frames)]
    }
    with open(f"{output_prefix}_sync.json", 'w') as f:
        json.dump(sync_data, f, indent=2)

    print(f"\n✅ 完成: {output_prefix}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        v_path  = sys.argv[1]
        out_dir = sys.argv[2]
        s_id    = Path(v_path).stem

        audio_dir = os.path.join(os.path.dirname(os.path.dirname(v_path)), "audio")
        a_path    = os.path.join(audio_dir, f"{s_id}.wav")
        if not os.path.exists(a_path):
            a_path = os.path.join(audio_dir, f"{s_id}_mix.wav")

        if not os.path.exists(a_path):
            print(f"❌ 找不到音訊: {a_path}")
            sys.exit(1)

        video_audio_process(v_path, a_path, output_prefix=os.path.join(out_dir, s_id))
    else:
        print("用法: python process_video.py [影片路徑] [輸出目錄]")
