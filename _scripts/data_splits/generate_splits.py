# generate_splits.py
# 환자 단위 train/val/test split 생성
import os
import random
from pathlib import Path

random.seed(42)

# CE-CT (Pretrain) 환자 폴더
ce_ct_root = Path("F:/LD-CT SR/Data/HCC CE-CT (D)")
ce_patients = sorted([d.name.split("_")[0] for d in ce_ct_root.iterdir() if d.is_dir()])

# NC-CT (Finetune) NIfTI 파일
nc_ct_root = Path("F:/LD-CT SR/Data/NC-CT NIfTI")
nc_patients = sorted([f.stem for f in nc_ct_root.glob("*.nii")])

print(f"CE-CT patients: {len(ce_patients)}")
print(f"NC-CT patients: {len(nc_patients)}")

# Split 비율: train 80%, val 10%, test 10%
def split_patients(patients, train_ratio=0.8, val_ratio=0.1):
    shuffled = patients.copy()
    random.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train:n_train+n_val],
        "test": shuffled[n_train+n_val:]
    }

# CE-CT split
ce_split = split_patients(ce_patients)
# NC-CT split
nc_split = split_patients(nc_patients)

# 저장
out_dir = Path(__file__).parent
out_dir.mkdir(exist_ok=True)

for name, splits in [("ce_ct", ce_split), ("nc_ct", nc_split)]:
    for mode, patients in splits.items():
        out_path = out_dir / f"{name}_{mode}.txt"
        with open(out_path, "w") as f:
            for p in patients:
                f.write(p + "\n")
        print(f"Saved: {out_path} ({len(patients)} patients)")

print("\nDone!")
