import os
import numpy as np
import h5py
from scipy.io import savemat

def crop_image(image, patch_size=512):
    """
    裁剪图像为指定大小的补丁
    """
    patches = []
    h, w = image.shape
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = image[i:i + patch_size, j:j + patch_size]
            if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
                patches.append(patch)
    return patches

def process_mat_files(input_folder, output_folder, patch_size=512):
    """
    处理文件夹中的所有 .mat 文件，裁剪图像为补丁并保存每个补丁为单独的文件
    """
    # 获取文件夹中所有的 .mat 文件
    mat_files = [f for f in os.listdir(input_folder) if f.endswith('.mat')]
    
    for mat_file in mat_files:
        # 构建完整路径
        mat_file_path = os.path.join(input_folder, mat_file)
        
        # 打开 .mat 文件
        with h5py.File(mat_file_path, 'r') as img:
            Inoisy = np.float32(np.array(img['Inoisy']).T)  # 加载数据
            
            # 生成补丁
            patches = crop_image(Inoisy, patch_size)
            
            # 保存每个补丁为单独的文件
            for idx, patch in enumerate(patches):
                # 构建输出文件名
                patch_filename = f"{os.path.splitext(mat_file)[0]}_patch_{idx+1}.mat"
                output_file_path = os.path.join(output_folder, patch_filename)
                
                # 将补丁保存为 .mat 文件
                output_data = {'Inoisy_patch': patch}
                savemat(output_file_path, output_data)
                print(f"Processed patch {idx+1} of {mat_file} and saved to {output_file_path}")

# 输入文件夹和输出文件夹路径
input_folder = '/data/lhq23/datasets/dnd/images_raw/'
output_folder = '/data/lhq23/datasets/dnd/images_raw/crops/'

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 处理文件
process_mat_files(input_folder, output_folder)
