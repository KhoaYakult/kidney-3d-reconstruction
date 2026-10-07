import SimpleITK as sitk # Dùng để đọc file y tế (.nii, .nii.gz, .dicom)
import numpy as np
import trimesh # Dùng để Post-Proccesing
from skimage import measure # Để sử dụng thuật toán Marching Cubes
import plotly.graph_objects as go
import scipy.ndimage as ndimage # Dùng để làm mờ ảnh (Gaussian Smoothing) giúp bề mặt mịn hơn 
import os
import sys
import webbrowser


# ==========================================
#  SỬA ĐƯỜNG DẪN CỦA BẠN Ở ĐÂY 
# ==========================================

# Bước 1: Tìm file 13.nii.gz 
# Bước 2: Chuột phải vào file đó -> Chọn "Copy as Path" 
# Bước 3: Dán vào bên dưới 

duong_dan_file = sys.argv[1] if len(sys.argv) > 1 else "path/to/case_segmentation.nii.gz"  # usage: python marching_cubes_plotly.py <mask.nii.gz> 

def generate_high_quality_mesh_local( nii_path, output_name = None):
    
    base_name = os.path.basename(nii_path)  
    simple_name = base_name.replace(".nii.gz", "").replace(".nii", "") 
    
    # Nếu người dùng không truyền tên, tự tạo tên chuẩn
    if output_name is None:
        safe_output_name = f"KetQua_Reconstruct_{simple_name}"
    else:
        safe_output_name = output_name
    
    
    print(f"\n---ĐANG XỬ LÝ FILE: {os.path.basename(nii_path)}---")
    print(f"-Đường dẫn đầy đủ: {nii_path}-")
    
    if not os.path.exists(nii_path):
        print(f"LỖI: Không tìm thấy file! Bạn hãy kiểm tra lại đường dẫn.")
        return

    print("--- Đang đọc dữ liệu NIfTI... ----")
    try:
        img = sitk.ReadImage(nii_path)
        
        #Do SimpleITK đọc file theo thứ tự (z,y,x) nên lệnh .transpose(2,1,0) nó sẽ đảo trục lại cho đúng 
        vol_data = sitk.GetArrayFromImage(img).transpose(2, 1, 0) 
        
        #Lấy kích thước thực tế của pixel
        spacing = img.GetSpacing()
        
    except Exception as e:
        print(f"=> Lỗi khi đọc file SimpleITK: {e}")
        return

    #Sau khi segmatation xong thì chúng ta sẽ có từng các số trùng lặp với nhau tượng trưng cho các điểm 
    #ảnh cấu tạo nên bộ phận đó
    
    unique_labels = np.unique(vol_data)
    print(f"Các nhãn tìm thấy trong ảnh: {unique_labels}")
    
    #Khởi tạo ra để chứa những bộ phận đã dựng xong
    meshes = []
    
    # BẢNG MÀU & TÊN (KiPA Dataset Standard)
    color_map = {
        1: [65, 105, 225, 255],   # Tĩnh mạch: Xanh Royal (Sang hơn xanh đậm)
        2: [210, 180, 140, 255],  # Thận: Màu Tan/Da người (Thay vì màu Vàng chói)
        3: [220, 20, 60, 255],    # Động mạch: Đỏ thẫm
        4: [50, 205, 50, 255]     # KHỐI U: Màu Xanh Lime (Để tương phản cực mạnh với màu thịt)
    }
    
    name_map = {
        1: "Tĩnh mạch (Vein)",
        2: "Quả Thận (Kidney)",  
        3: "Động mạch (Artery)", 
        4: "Khối U (Tumor)"      
    }
    
    scene = trimesh.Scene()

    
    for label in unique_labels:
        if label == 0: 
            continue # Bỏ qua nền đen
        
        display_name = name_map.get(label, f"Label lại {label}")
        print(f"   -> Đang dựng hình 3D: {display_name}...")
        
        #BƯỚC 1: Tiền xử lý
        
        #Tách riêng đối tượng cần xử lý ra ( Nếu đúng là bộ phận cần tính thì gán 1, không thì gán 0)
        binary_mask = (vol_data == label).astype(float)
        
        #Gaussian Smoothing (Làm mượt pixel răng cưa)
        #Giảm sigma xuống 1.0 để giữ chi tiết khối u tốt hơn
        #smooth_mask = ndimage.gaussian_filter(binary_mask, sigma = 1.0) 
        
        if label in [1, 3]: # Nếu là Tĩnh mạch hoặc Động mạch
            print(" -> Đang vá mạch máu bị đứt...")

            # A. Dùng 'Phép đóng' (Closing) để nối các đoạn đứt nhỏ
            # Tạo một cấu trúc hình cầu nhỏ để quét
            struct = ndimage.generate_binary_structure(3, 1) 
            # Lặp 2 lần để nối các khe hở khoảng 1-2 pixel
            binary_mask = ndimage.binary_closing(binary_mask, structure = struct, iterations = 2).astype(float)
            
            # B. Giảm Gaussian xuống thấp để không làm mất mạch nhỏ
            current_sigma = 0.5 
            
        else: # Nếu là Thận hoặc U (Khối lớn)
            # Giữ nguyên cấu hình cũ cho đẹp
            current_sigma = 1.0

        # 2. Làm mềm với sigma đã chọn
        smooth_mask = ndimage.gaussian_filter(binary_mask, sigma = current_sigma)
        
        #BƯỚC 2: Dựng bằng thuật toán marching_cubes
        
        try:
            verts, faces, normals, values = measure.marching_cubes(
                smooth_mask, level = 0.5, spacing = spacing, step_size = 1, method = 'lewiner' ) #level -> ngưỡng iso-value
            
        except Exception as e:
            print(f"--- Không dựng được {display_name} (có thể quá nhỏ): {e}---")
            continue

        #BƯỚC 3: Hậu xử lý
        #Mesh giúp gôm các đỉnh và các mặt tạo thành một object
        mesh = trimesh.Trimesh(vertices = verts, faces = faces)
        
        #Dòng quan trọng 
        #Thuật toán Taubin vừa mài mòn các gai nhọn (nhiễu) nhưng lại đẩy phồng mô hình ra lại để giữ nguyên thể tích khối u
        trimesh.smoothing.filter_taubin(mesh, iterations = 10) # Giảm iteration để đỡ bị teo hình
        
        # Gán màu
        mesh.visual.face_colors = color_map.get(label, [128, 128, 128, 255])
        
        scene.add_geometry(mesh)
        meshes.append((label, mesh, display_name))

    if len(meshes) > 0:
        print("-> Đang vẽ đồ họa Plotly...")
        fig = go.Figure()
        
        for label, mesh, name_str in meshes:
            v = mesh.vertices
            f = mesh.faces
            c = color_map.get(label, [128, 128, 128, 255])
            color_str = f'rgb({c[0]},{c[1]},{c[2]})'
            
            # Chỉnh độ trong suốt: Thận mờ (0.3) để nhìn xuyên thấy U (1.0)
            opacity_val = 0.7 if label == 2 else 1.0 
            
            fig.add_trace(go.Mesh3d(
                x = v[:,0], y = v[:,1], z = v[:,2],
                i = f[:,0], j = f[:,1], k = f[:,2],
                color = color_str,
                opacity = opacity_val, 
                name = name_str, 
                flatshading = False,
                lighting = dict(ambient = 0.5, diffuse = 0.8, roughness = 0.5, specular = 0.1, fresnel = 0.5)
            ))
            
        fig.update_layout(
            title = f"Kết quả tái cấu trúc sau khi phân đoạn 3D: {os.path.basename(nii_path)}",
            scene = dict(
                xaxis = dict(visible=False),
                yaxis = dict(visible=False),
                zaxis = dict(visible=False),
                aspectmode = 'data'
            ),
            margin = dict( l = 0, r = 0, b = 0, t = 40)
        )
        
        # Xuất file HTML với tên an toàn
        print("*"*30)
        html_file = f"{safe_output_name}.html" # Dùng tên đã xử lý sạch
        full_html_path = os.path.abspath(html_file)
        
        try:
            fig.write_html(html_file)
            print("\n" + "="*40)
            print(f"XONG! Đã lưu file HTML tại:")
            print(f" {full_html_path}")
            print("="*40)
            
            # Tự động mở trình duyệt
            webbrowser.open('file://' + full_html_path)
        except Exception as e:
            print(f"Lỗi khi lưu file HTML: {e}")

    else:
        print("---Không tạo được mô hình nào (File rỗng?).---")


generate_high_quality_mesh_local(duong_dan_file)