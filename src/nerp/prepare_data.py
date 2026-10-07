import nibabel as nib
import numpy as np
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))

# 1. Cấu hình
input_nii_path = sys.argv[1] if len(sys.argv) > 1 else "path/to/case_segmentation.nii.gz"  # usage: python prepare_data.py <mask.nii.gz>

# NeRP yêu cầu cấu trúc thư mục data/ct_data
output_folder = os.path.join(current_dir,"data", "ct_data")
os.makedirs(output_folder, exist_ok=True)

# 2. Đọc file NIfTI
print(f"Dang doc file: {input_nii_path}")
try:
    nii = nib.load(input_nii_path)
    vol_data = nii.get_fdata()
except FileNotFoundError:
    print(f"Loi: Khong tim thay file tai {input_nii_path}")
    exit()

# 3. Chuẩn hóa dữ liệu về khoảng [0, 1]
print("Dang chuan hoa du lieu...")
vol_data = (vol_data - vol_data.min()) / (vol_data.max() - vol_data.min())

# 4. Lưu thành .npy để NeRP đọc
save_path = os.path.join(output_folder, "kidney_13.npy")
np.save(save_path, vol_data)

print(f"Da xong! File du lieu cho NeRP: {save_path}")
print(f"Kich thuoc file la: {vol_data.shape}")