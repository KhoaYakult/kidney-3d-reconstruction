import nibabel as nib
import numpy as np
import os
import cv2

# --- CẤU HÌNH ---
input_nii_path = r"13.nii.gz"  # Đảm bảo file này nằm cùng thư mục code
output_folder = os.path.join("data", "ct_data")
os.makedirs(output_folder, exist_ok=True)
save_path = os.path.join(output_folder, "kidney_13.npz")

# 1. Đọc file NIfTI
print(f"Dang doc file: {input_nii_path}")
try:
    nii = nib.load(input_nii_path)
    vol_data = nii.get_fdata()
except FileNotFoundError:
    print(f"Loi: Khong tim thay file {input_nii_path}")
    exit()

# 2. Chuẩn hóa về [0, 1]
print("Dang chuan hoa du lieu...")
vol_data = (vol_data - vol_data.min()) / (vol_data.max() - vol_data.min())

# 3. Resize nếu cần (Optional) - NeRP thích hình vuông
# Code gốc thường crop hoặc pad để các chiều x, y bằng nhau.
# Ở đây ta giữ nguyên, để code data.py tự xử lý crop/pad.

# 4. Lưu dưới dạng .npz nén (Compressed) với key là 'data'
print(f"Dang luu file .npz tai: {save_path}")
np.savez_compressed(save_path, data=vol_data)

print(f"XONG! Kich thuoc du lieu: {vol_data.shape}")
print("Hay dung file nay cho config.")