
import json 
from tqdm import tqdm
import numpy as np

# Đặt các giá trị ngẫu nhiên
n_timesteps = 10  # Số bước thời gian (kích thước thứ 1)
n_features = 5    # Số lượng đặc trưng (kích thước thứ 2)
n_samples = 100   # Số mẫu (kích thước thứ 3)

# Tạo mảng ngẫu nhiên cho traindatasets_valuesF
# Giả sử values là các giá trị liên tục
traindatasets_valuesF = np.random.rand(n_timesteps, n_features, n_samples)

# Tạo mảng ngẫu nhiên cho traindatasets_maskF (0 hoặc 1)
# Giả sử mask là các giá trị nhị phân
traindatasets_maskF = np.random.randint(0, 2, size=(n_timesteps, n_samples))

# Tạo mảng ngẫu nhiên cho traindatasets_evalmaskF (0 hoặc 1)
# Giả sử eval_mask là các giá trị nhị phân
traindatasets_evalmaskF = np.random.randint(0, 2, size=(n_timesteps, n_samples))

# Tạo mảng ngẫu nhiên cho traindatasets_deltaF
# Giả sử delta là các giá trị thời gian giữa các sự kiện
traindatasets_deltaF = np.random.rand(n_timesteps, n_features, n_samples)

# Tạo mảng ngẫu nhiên cho traindatasets_deltaBF (có thể có cùng kích thước với deltaF)
# Giả sử deltaB là các giá trị thời gian ngược
traindatasets_deltaBF = np.random.rand(n_timesteps, n_features, n_samples)

# In ra kích thước của các mảng
print('traindatasets_valuesF shape:', traindatasets_valuesF.shape)
print('traindatasets_maskF shape:', traindatasets_maskF.shape)
print('traindatasets_evalmaskF shape:', traindatasets_evalmaskF.shape)
print('traindatasets_deltaF shape:', traindatasets_deltaF.shape)
print('traindatasets_deltaBF shape:', traindatasets_deltaBF.shape)


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

all_len = traindatasets_valuesF.shape[2]
fs = open('train.json', 'w')
print('save training dataset as json: ')
for id_ in tqdm(range(all_len)):
    parse_idTrain(id_)

del traindatasets_valuesF,traindatasets_evalmaskF,traindatasets_maskF,traindatasets_deltaF,traindatasets_deltaBF
fs.close()
