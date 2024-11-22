# import numpy as np
# import matplotlib.pyplot as plt

# ndvi= np.load('/mnt/storage/huyekgis/brios/data2/ndvi_timeseries.npy')

# ndvi_root = ndvi.copy()

# x, y, t = ndvi.shape

# ndvi = ndvi.reshape(x * y, t)

# first_index = 942

# print(f"first dnvi: {ndvi[first_index, 10]}")

# index_for_train = np.array([1, 3, 5, 7, 942])

# print(x * y)

# coordinates_first = np.array([(i, j) for i in range(x) for j in range(y)])

# test = np.load('/mnt/storage/huyekgis/brios/BRIOS/models/inference_results.npy')

# print(f"test shape: {test.shape}")

# print(f"coor first: {coordinates_first[942]}")

# coordinates_second = coordinates_first[index_for_train]

# second_ndvi = ndvi[index_for_train]

# second_index = 4
# print(f"second ndvi: {second_ndvi[second_index, 10]}")
# print(f"coor second: {coordinates_second[4]}")

# x, y = coordinates_second[4]
# t = 10

# ndvi_image = ndvi_root[:, :, t]

# hiephoa = np.load('/mnt/storage/huyekgis/brios/BRIOS/models/inference_results.npy')
# print(hiephoa)

import numpy as np 

# t = np.load('/mnt/storage/huyekgis/brios/BRIOS/models/test.npy')

# print(t[0])

import torch

# Load file .pt
checkpoint = torch.load("/mnt/storage/huyekgis/brios/BRIOS/models/model_file/biros_base/base_brios.pt")

# Kiểm tra các key trong checkpoint
print("Keys in checkpoint:", checkpoint.keys())

# In chi tiết từng key
for key, value in checkpoint.items():
    print(f"Key: {key}, Type: {type(value)}")

    # Nếu là model_state_dict, in một số layer
    if key == 'model_state_dict':
        print("Example layers in model_state_dict:")
        for layer, weights in list(value.items())[:5]:  # In 5 layer đầu
            print(f"Layer: {layer}, Weights shape: {weights.shape}")
