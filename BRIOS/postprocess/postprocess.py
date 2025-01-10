import os
import re
import rasterio
import numpy as np
import tifffile as tiff
from tqdm import tqdm

folder_path = '/mnt/storage/code/EOV_NDVI/brios/datasets/Train_LST/ThuyVan-ThaiThuy-ThaiBinh'
npy_path = '/mnt/storage/code/EOV_NDVI/brios/datasets/Train_LST/ThuyVan-ThaiThuy-ThaiBinh/infer_thuyvan.npy'

def npy_to_tiff(folder_path, npy_path):
    img1 = np.load(npy_path)
    print(img1.shape)
    rvi_path = os.path.join(folder_path, 'thuyvan_rvi_8days')
    ndvi_path = os.path.join(folder_path, 'thuyvan_ndvi8days')
    tif_files = [f for f in os.listdir(rvi_path) if f.endswith('.tif')]

    dates = []
    for tif_file in tqdm(tif_files):
        match = re.search(r'(\d{4}-\d{2}-\d{2})', tif_file)  # Tìm ngày theo định dạng 'YYYY-MM-DD'
        if match:
            dates.append(match.group(1))  # Thêm ngày vào danh sách

    # Sắp xếp danh sách ngày theo thứ tự tăng dần
    dates.sort()
    list_ndvi_path = os.listdir(ndvi_path)
    # Đọc metadata từ file gốc
    with rasterio.open(f"{ndvi_path}/{list_ndvi_path[0]}") as src:
        meta = src.meta

    # Giả sử ndvi_root là mảng NDVI với shape (x, y, t)
    num_timesteps = 46

    # Kiểm tra số lượng ngày có sẵn
    assert len(dates) == num_timesteps, "Số ngày không khớp với số thời điểm"

    # Lặp qua từng thời điểm và lưu các ảnh với tên tệp dựa trên thứ tự ngày
    for t in range(num_timesteps):
        # Tạo tên file cho mỗi ảnh dựa trên ngày
        filename = f"/mnt/storage/code/EOV_NDVI/brios/datasets/Train_LST/NDVI_ThuyVan/img_infer_thuyvan_{dates[t]}.tif"
        
        # Sử dụng metadata từ file gốc để lưu lại thông tin metadata cho file mới
        meta.update(count=1, dtype=rasterio.float32)  # Cập nhật số lượng band và kiểu dữ liệu

        # Lưu ảnh dự đoán vào file mới với metadata sao chép
        with rasterio.open(filename, 'w', **meta) as dst:
            dst.write(img1[:, :, t], 1)  # Ghi ảnh vào band đầu tiên


npy_to_tiff(folder_path, npy_path)