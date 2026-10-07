"""
Script này thay thế thuật toán Marching Cubes bằng NeRP/SIREN
để tạo mesh 3D chất lượng cao từ medical images.

Ưu điểm:
- Siêu phân giải (query ở độ phân giải tùy ý)
- Bề mặt mịn tự nhiên (không cần Gaussian smoothing)
- Nội suy chính xác giữa các lát cắt
"""

import torch
import numpy as np
from networks import SIREN, Positional_Encoder
from skimage import measure
import trimesh
import plotly.graph_objects as go
import yaml
import os
import webbrowser

# ==========================================
#  CẤU HÌNH
# ==========================================
CHECKPOINT_PATH = 'outputs/kidney_result/checkpoints/model_25000.pt'
CONFIG_PATH = 'outputs/kidney_result/config.yaml'
OUTPUT_NAME = 'NeRP_Reconstruction_Kidney13'

# Độ phân giải reconstruction (cao hơn = chi tiết hơn nhưng chậm hơn)
RESOLUTION = 512  # Thử 128, 256, 384, 512

# Ngưỡng iso-surface (điều chỉnh để lấy đúng vùng cần thiết)
ISO_THRESHOLD = 0.5  # 0.0-1.0

# ==========================================
#  HÀM CHÍNH
# ==========================================

def nerp_to_mesh(checkpoint_path, config_path, resolution=256, iso_threshold=0.5):
    """
    Chuyển đổi NeRP model thành mesh 3D
    
    Args:
        checkpoint_path: Đường dẫn đến model đã train
        config_path: Đường dẫn đến file config
        resolution: Độ phân giải sampling (càng cao càng chi tiết)
        iso_threshold: Ngưỡng để tạo bề mặt
    
    Returns:
        trimesh.Trimesh: Mesh 3D
    """
    print("="*50)
    print("🧠 NeRP-BASED 3D RECONSTRUCTION")
    print("="*50)
    
    # Load config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    # Load model
    print(f"📥 Loading model from: {checkpoint_path}")
    encoder = Positional_Encoder(config['encoder'])
    model = SIREN(config['net']).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    if checkpoint['enc'] is not None:
        encoder.B = checkpoint['enc'].to(device)
    
    model.eval()
    print(f"✅ Model loaded successfully (Loss: {checkpoint.get('loss', 'N/A')})")
    
    # Tạo grid 3D với độ phân giải cao
    print(f"🔍 Sampling volume at resolution: {resolution}³")
    coords = torch.linspace(0, 1, resolution)
    grid_z, grid_y, grid_x = torch.meshgrid(coords, coords, coords, indexing='ij')
    grid = torch.stack([grid_z, grid_y, grid_x], dim=-1)
    flat_grid = grid.reshape(-1, 3)
    
    # Reconstruction
    print("🎨 Reconstructing volume...")
    batch_size = 8192
    reconstructed = []
    
    with torch.no_grad():
        for i in range(0, flat_grid.shape[0], batch_size):
            batch = flat_grid[i:i+batch_size].to(device)
            pred = model(encoder.embedding(batch))
            reconstructed.append(pred.cpu())
            
            if i % (batch_size * 10) == 0:
                progress = (i / flat_grid.shape[0]) * 100
                print(f"   Progress: {progress:.1f}%")
    
    reconstructed = torch.cat(reconstructed, dim=0)
    volume = reconstructed.reshape(resolution, resolution, resolution).numpy()
    
    print(f"✅ Volume reconstructed: shape={volume.shape}, range=[{volume.min():.3f}, {volume.max():.3f}]")
    
    # Apply Marching Cubes (nhưng trên volume chất lượng cao từ NeRP!)
    print(f"🔺 Extracting mesh using Marching Cubes (threshold={iso_threshold})...")
    
    try:
        verts, faces, normals, values = measure.marching_cubes(
            volume, 
            level=iso_threshold,
            spacing=(1.0/resolution, 1.0/resolution, 1.0/resolution),
            step_size=1,
            method='lewiner'
        )
    except Exception as e:
        print(f"❌ Marching Cubes failed: {e}")
        print("💡 Try adjusting ISO_THRESHOLD parameter")
        return None
    
    # Tạo mesh
    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals)
    
    print(f"✅ Mesh extracted:")
    print(f"   - Vertices: {len(mesh.vertices):,}")
    print(f"   - Faces: {len(mesh.faces):,}")
    print(f"   - Watertight: {mesh.is_watertight}")
    
    # Optional: Smoothing (NeRP đã mịn rồi nên không cần nhiều)
    print("🎨 Applying light smoothing...")
    trimesh.smoothing.filter_taubin(mesh, iterations=5)
    
    return mesh


def visualize_mesh_plotly(mesh, output_html='output.html', title='NeRP 3D Reconstruction'):
    """
    Trực quan hóa mesh bằng Plotly
    
    Args:
        mesh: trimesh.Trimesh object
        output_html: Tên file HTML output
        title: Tiêu đề
    """
    print("🎨 Creating interactive visualization...")
    
    v = mesh.vertices
    f = mesh.faces
    
    # Tính màu dựa trên độ cao (z-coordinate)
    z_min, z_max = v[:, 2].min(), v[:, 2].max()
    colors = (v[:, 2] - z_min) / (z_max - z_min)  # Normalize to [0, 1]
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=v[:, 0],
            y=v[:, 1],
            z=v[:, 2],
            i=f[:, 0],
            j=f[:, 1],
            k=f[:, 2],
            intensity=colors,
            colorscale='Viridis',
            opacity=0.8,
            name='Kidney',
            flatshading=False,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                roughness=0.5,
                specular=0.2,
                fresnel=0.5
            )
        )
    ])
    
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    
    # Lưu file
    full_path = os.path.abspath(output_html)
    fig.write_html(output_html)
    
    print("="*50)
    print(f"✅ DONE! HTML saved at:")
    print(f"👉 {full_path}")
    print("="*50)
    
    # Tự động mở trình duyệt
    webbrowser.open('file://' + full_path)


def export_formats(mesh, base_name='output'):
    """
    Export mesh sang nhiều định dạng khác nhau
    
    Args:
        mesh: trimesh.Trimesh object
        base_name: Tên file cơ bản (không có extension)
    """
    print("\n📦 Exporting to multiple formats...")
    
    formats = {
        'obj': 'Wavefront OBJ (Blender, Maya)',
        'stl': 'STL (3D printing)',
        'ply': 'PLY (MeshLab)',
        'off': 'OFF (Geomview)',
        'glb': 'GLB (Web/AR/VR)'
    }
    
    for ext, desc in formats.items():
        try:
            filename = f"{base_name}.{ext}"
            mesh.export(filename)
            print(f"   ✅ {filename} - {desc}")
        except Exception as e:
            print(f"   ❌ Failed to export {ext}: {e}")


# ==========================================
#  MAIN EXECUTION
# ==========================================

if __name__ == '__main__':
    # Bước 1: Tạo mesh từ NeRP
    mesh = nerp_to_mesh(
        CHECKPOINT_PATH, 
        CONFIG_PATH, 
        resolution=RESOLUTION,
        iso_threshold=ISO_THRESHOLD
    )
    
    if mesh is None:
        print("❌ Failed to generate mesh. Exiting...")
        exit()
    
    # Bước 2: Trực quan hóa
    visualize_mesh_plotly(
        mesh, 
        output_html=f'{OUTPUT_NAME}.html',
        title='NeRP-based 3D Reconstruction (SOTA)'
    )
    
    # Bước 3: Export sang nhiều định dạng
    export_formats(mesh, base_name=OUTPUT_NAME)
    
    print("\n🎉 ALL DONE!")
    print("💡 Next steps:")
    print("   - Open HTML file to view interactive 3D model")
    print("   - Import .obj/.stl files into Blender/3D Slicer")
    print("   - Adjust ISO_THRESHOLD if shape looks wrong")