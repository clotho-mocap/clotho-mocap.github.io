"""Export GT / prediction windows for the project-page viewer.

The source is the Stage Tran viewer export. Each clip keeps the ground-truth
root and joints plus the ``online_guarded`` prediction, from 2 s through 12 s.
"""

import json
import shutil
import struct
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "FGP" / "stage_tran" / "viewer"
DEST = ROOT / "qualitative"
START_S = 2
DURATION_S = 10

CLIPS = [
    ("43_180014_zzx_bag", "Daily", "Bag"),
    ("36_173925_zzx_walk", "Daily", "Walk"),
    ("05_181400_hlj_golf", "Sports", "Golf"),
    ("10_170315_lc_basketball", "Sports", "Basketball"),
    ("12_170711_lc_longJump", "Sports", "Long Jump"),
    ("23_151719_zkb_badminton", "Sports", "Badminton"),
    ("33_192238_yy_warmUp", "Fitness", "Warm Up"),
    ("41_175649_zzx_box", "Fitness", "Boxing"),
    ("25_152104_zkb_gymnastics", "Fitness", "Gymnastics"),
    ("45_180404_zzx_tie_shoelace", "Daily", "Tie Shoelaces"),
    ("27_191100_yy_sit", "Daily", "Sit"),
    ("14_171053_lc_clean", "Daily", "Clean"),
    ("26_152302_zkb_bowStep", "Fitness", "Bow Step"),
]


def load_array(raw, meta, key):
    """Return one packed float array from a viewer binary."""
    spec = meta["arrays"][key]
    offset = spec["offset"] // 4
    count = int(np.prod(spec["shape"]))
    return raw[offset:offset + count].reshape(spec["shape"])


def main():
    clip_dir = DEST / "clips"
    clip_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(SOURCE / "web" / "smpl.bin", DEST / "smpl.bin")
    manifest = {"fps": 30, "prediction": "online_guarded", "clips": []}
    for source_id, category, title in CLIPS:
        meta = json.loads((SOURCE / "data" / f"{source_id}.json").read_text(encoding="utf-8"))
        raw = np.fromfile(SOURCE / "data" / f"{source_id}.bin", dtype="<f4")
        gt_root = load_array(raw, meta, "gt_root")
        gt_joints = load_array(raw, meta, "gt_joints")
        pred_root = load_array(raw, meta, "root_online_guarded")
        pred_joints = load_array(raw, meta, "joints_online_guarded")
        fps = float(meta["fps"])
        start = int(round(START_S * fps))
        count = int(round(DURATION_S * fps))
        if start + count > gt_root.shape[0]:
            raise RuntimeError(f"{source_id} is shorter than {START_S + DURATION_S:.0f}s")
        end = start + count
        gt_root = np.ascontiguousarray(gt_root[start:end], dtype="<f4")
        gt_joints = np.ascontiguousarray(gt_joints[start:end], dtype="<f4")
        pred_root = np.ascontiguousarray(pred_root[start:end], dtype="<f4")
        pred_joints = np.ascontiguousarray(pred_joints[start:end], dtype="<f4")
        feet = gt_joints[:, [10, 11], 1] + gt_root[:, None, 1]
        reference_y = float(np.quantile(feet, 0.05))
        slug = title.lower().replace(" ", "-")
        blob = struct.pack("<4sIf", b"QLIP", count, np.float32(reference_y))
        payload = np.concatenate([
            gt_root.reshape(-1),
            pred_root.reshape(-1),
            gt_joints.reshape(-1),
            pred_joints.reshape(-1),
        ]).astype("<f4").tobytes()
        (clip_dir / f"{slug}.bin").write_bytes(blob + payload)
        manifest["clips"].append({
            "id": slug,
            "source": source_id,
            "category": category,
            "title": title,
            "file": f"clips/{slug}.bin",
            "sourceStart": start,
            "frames": count,
        })
        print(f"{title:16} {source_id}  start={start:4d}  frames={count}")
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
