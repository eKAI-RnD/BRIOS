import numpy as np
import json
from tqdm import tqdm

# Random data generation function
def generate_random_data(shape):
    """Generates random data to simulate GEO TIFF data."""
    return np.random.rand(*shape)

# Giả sử chúng ta có các tham số sau
rows = 1000  # số hàng của ảnh (tùy chỉnh)
cols = 100  # số cột của ảnh (tùy chỉnh)
n_bands = 46  # số băng của ảnh (tùy chỉnh)
n_samples = rows * cols

# Sinh dữ liệu ngẫu nhiên thay thế cho các file đọc từ tệp
areamask = generate_random_data((rows, cols))
print(f'area mask shape: {areamask.shape}')
cloudmask = np.random.randint(0, 3, size=(rows, cols, n_bands))  # Giá trị ngẫu nhiên 0, 1, 2 cho các mặt nạ mây
print(f'cloudmask shape: {cloudmask.shape}')
datanum = np.apply_along_axis(lambda x: np.count_nonzero(np.logical_not(x)), axis=2, arr=cloudmask)

# Chia dữ liệu huấn luyện ngẫu nhiên
areamask0 = areamask.reshape((n_samples))
print(f'area mask 0 shape: {areamask0.shape}')
datanum0 = datanum.reshape((n_samples))
idx = np.argwhere((areamask0 != 0) & (datanum0 > 25))
train_index = np.random.choice(idx.flatten(), size=len(idx), replace=False)

# Tiếp tục sinh các dữ liệu ngẫu nhiên khác cho từng bước
trainmask = np.zeros(n_samples)
trainmask[train_index] = 1
trainmask = trainmask.reshape((rows, cols))

# Sinh các dữ liệu ngẫu nhiên cho từng batch
feature_num = 3
time = np.arange(1, 736, 16)

# Khởi tạo các mảng dữ liệu huấn luyện
traindatasets_valuesF = np.random.rand(n_bands, feature_num, 10000).astype(np.float16)
traindatasets_evalmaskF = np.random.randint(0, 2, (n_bands, 10000)).astype(np.int8)
traindatasets_maskF = np.random.randint(0, 2, (n_bands, 10000)).astype(np.int8)
traindatasets_deltaF = np.random.rand(n_bands, feature_num, 10000).astype(np.float16)
traindatasets_deltaBF = np.random.rand(n_bands, feature_num, 10000).astype(np.float16)

# Tạo JSON ngẫu nhiên
def parse_rec(values, masks, eval_masks, deltas):
    rec = []
    for i in range(values.shape[0]):
        recone = {}
        recone['deltas'] = deltas[i, :].tolist()
        recone['masks'] = masks[i].astype('int8').tolist()
        recone['values'] = values[i, :].tolist()
        recone['eval_masks'] = eval_masks[i].astype('int8').tolist()
        rec.append(recone)
    return rec

def parse_idTrain(id_):
    values = traindatasets_valuesF[:, :, id_]
    masks = traindatasets_maskF[:, id_]
    eval_masks = traindatasets_evalmaskF[:, id_]
    deltas = traindatasets_deltaF[:, :, id_]
    deltasB = traindatasets_deltaBF[:, :, id_]

    rec = {}

    rec['forward'] = parse_rec(values, masks, eval_masks, deltas)
    rec['backward'] = parse_rec(values[::-1], masks[::-1], eval_masks[::-1], deltasB)

    rec = json.dumps(rec)
    fs.write(rec + '\n')

# Lưu dữ liệu ngẫu nhiên dưới dạng JSON
fs = open('random_training_data.json', 'w')
all_len = traindatasets_valuesF.shape[2]
print('Save random training dataset as JSON: ')
for id_ in tqdm(range(all_len)):
    parse_idTrain(id_)
fs.close()
