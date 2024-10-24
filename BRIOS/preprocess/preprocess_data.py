import numpy as np
import json
from tqdm import tqdm





# Cac function can dung
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

def cal_timestep(time, mask):
    """calculate timestep (between step t to step t-i nearest without cloud)

    Args:
        time (_type_): _description_
        mask (_type_): _description_

    Returns:
        deltaT: timestep
    """
    deltaT = time.copy()
    for i in range(len(time)):
        T_time0 = time[i]
        if i != 0:
            for k in range(i - 1, -1, -1):
                T_time1 = time[k]
                if mask[k] == 1:
                    T_time1 = time[k]
                    break

            T = T_time0-T_time1
        else:
            T = 0

        deltaT[i] = T

    return deltaT


def generate_random_data(shape):
    """Generates random data to simulate GEO TIFF data."""
    return np.random.rand(*shape)






# Parameters
# rows = 10  # number of rows in the image
# cols = 10  # number of columns in the image
n_bands = 46  # number of bands in the image
n_samples = 100
# n_samples = rows * cols


"""
Model yêu cầu 5 features: cloudmask, areamask, ndvi, rvi, vh
Vì ban đầu đã fill hết mây nên set up cloudmask = 0 (không có mây) và areamask = 1 (có data)
"""



# Generate the cloudmask where all values are 0 (clear observations, no clouds)
cloudmask = np.zeros((n_samples, n_bands), dtype=np.int8)  # All clear, no clouds
print(f'cloudmask shape: {cloudmask.shape}')

# Generate the areamask based on the condition (0-invalid/no data if cloudmask == 1 or 2, otherwise 1-valid)
# Since all cloudmask values are 0 (clear), all values in areamask will be 1 (valid)
areamask = np.ones(n_samples, dtype=np.int8)  # All valid data
print(f'area mask shape: {areamask.shape}')

# Count the number of clear observations (cloudmask == 0) per pixel
datanum = np.full(n_samples, n_bands, dtype=np.int8)  # All pixels have all clear observations
print(f'Number of clear observations per pixel: {datanum[:10]} (showing first 10 samples)')

# Select indices for training where areamask is not 0 and there are more than 25 clear observations
idx = np.argwhere((areamask != 0) & (datanum > 25)).flatten()
train_index = np.random.choice(idx, size=len(idx), replace=False)

print(f'Number of selected training samples: {len(train_index)}')

# Generate the training mask based on selected training indices
trainmask = np.zeros(n_samples, dtype=np.int8)  # Flattened
trainmask[train_index] = 1  # Mark the selected training samples
print(f'trainmask shape: {trainmask.shape}')









# Sinh các dữ liệu ngẫu nhiên cho từng batch
feature_num = 3
time = np.arange(1, 736, 16)


# Khoi tao mang numpy de luu value của 5 features
traindatasets_valuesF = np.empty((n_bands, 3, 0),dtype=np.float16)
traindatasets_evalmaskF = np.empty((n_bands, 0),dtype=np.int8)
traindatasets_maskF = np.empty((n_bands, 0),dtype=np.int8)
traindatasets_deltaF = np.empty((n_bands, 3, 0),dtype=np.float16)
traindatasets_deltaBF = np.empty((n_bands, 3, 0),dtype=np.float16)



# ndvi, vh, rvi path
ndvi_data_path = '/mnt/data1tb/BRIOS/dataTrain/normNDVI_cut.npy'
vh_data_path = '/mnt/data1tb/BRIOS/dataTrain/normVH_cut.npy'
rvi_data_path = '/mnt/data1tb/BRIOS/dataTrain/normRVI_cut.npy'

# Load data from the .npy files
ndvi_data = np.load(ndvi_data_path)
vh_data = np.load(vh_data_path)
rvi_data = np.load(rvi_data_path)

# Print the shapes of the loaded data to verify
print(f'NDVI data shape: {ndvi_data.shape}')
print(f'VH data shape: {vh_data.shape}')
print(f'RVI data shape: {rvi_data.shape}')


# get index
ndvi_data0 = ndvi_data
vh_data0 = vh_data
rvi_data0 = rvi_data
cloudmask_arr = cloudmask
trainmask_arr = trainmask
train_index0 = np.where(trainmask_arr == 1)
train_index0 = train_index0[0]

# get data train
ndvi_dataT = ndvi_data0[train_index0, :]
vh_dataT = vh_data0[train_index0, :]
rvi_dataT = rvi_data0[train_index0, :]
cloudmask_arrT = cloudmask_arr[train_index0, :]
maskT =  np.where(cloudmask_arrT == 0, 1, 0)
eval_maskT = np.where(cloudmask_arrT == 2, 1, 0)

print(eval_maskT)


# gen time interval for training forward
print('Generate time interval for training dataset: ')
deltaT = np.zeros((len(train_index0), n_bands))
for i in tqdm(range(len(train_index0))):
    maskone = maskT[i, :]
    done = cal_timestep(time, maskone)
    deltaT[i, :] = done

# gen time interval for training backward
print('Generate backward time interval for training dataset: ')
deltaTb = np.zeros((len(train_index0), n_bands))
for i in tqdm(range(len(train_index0))):
    maskone = maskT[i, :]
    maskone = maskone[::-1]
    done = cal_timestep(time, maskone)
    deltaTb[i, :] = done

# gen time interval for SAR 
print('Generate time interval for SAR data: ')
deltaTt = np.zeros((len(train_index0), n_bands))
for i in tqdm(range(len(train_index0))):
    maskone = np.ones(n_bands)
    maskone = np.int_(maskone)
    done = cal_timestep(time, maskone)
    deltaTt[i, :] = done

# create training dataset
print('Generate training dataset: ')
traindatasets_values = np.zeros((n_bands, feature_num, len(train_index0)),dtype=np.float16)
for i in tqdm(range(n_bands)):
    for k in range(len(train_index0)):
        traindatasets_values[i, 0, k] = vh_dataT[k, i]
        traindatasets_values[i, 1, k] = rvi_dataT[k, i]
        traindatasets_values[i, 2, k] = ndvi_dataT[k, i]
traindatasets_valuesF = np.concatenate((traindatasets_valuesF, traindatasets_values), axis=2)

print('Generate evalmask: ')  #evalmask: where is the validation/simulated data used as evaluation
traindatasets_evalmask = np.zeros((n_bands, len(train_index0)),dtype=np.int8)
for i in tqdm(range(n_bands)):
    for k in range(len(train_index0)):
        traindatasets_evalmask[i, k] = eval_maskT[k, i]
traindatasets_evalmaskF = np.concatenate((traindatasets_evalmaskF, traindatasets_evalmask), axis=1)

print('Generate mask: ')   #mask: where is cloudy/missing data including real and simulated one
traindatasets_mask = np.zeros((n_bands, len(train_index0)),dtype=np.int8)
for i in tqdm(range(n_bands)):
    for k in range(len(train_index0)):
        traindatasets_mask[i, k] = maskT[k, i]
traindatasets_maskF = np.concatenate((traindatasets_maskF, traindatasets_mask), axis=1)
del eval_maskT,maskT

print('Generate training dataset: ')
traindatasets_delta = np.zeros((n_bands, feature_num, len(train_index0)),dtype=np.float16)
for i in tqdm(range(n_bands)):
    for k in range(len(train_index0)):
        traindatasets_delta[i, 0, k] = deltaTt[k, i]
        traindatasets_delta[i, 1, k] = deltaTt[k, i]
        traindatasets_delta[i, 2, k] = deltaT[k, i]
traindatasets_deltaF = np.concatenate((traindatasets_deltaF, traindatasets_delta), axis=2)


print('Generate training dataset (backward): ')
traindatasets_deltaB = np.zeros((n_bands, feature_num, len(train_index0)),dtype=np.float16)
for i in tqdm(range(n_bands)):
    for k in range(len(train_index0)):
        traindatasets_deltaB[i, 0, k] = deltaTt[k, i]
        traindatasets_deltaB[i, 1, k] = deltaTt[k, i]
        traindatasets_deltaB[i, 2, k] = deltaTb[k, i]
traindatasets_deltaBF = np.concatenate((traindatasets_deltaBF, traindatasets_deltaB), axis=2)









# # Khởi tạo các mảng dữ liệu huấn luyện
# traindatasets_valuesF = np.random.rand(n_bands, feature_num, 100).astype(np.float16)
# traindatasets_evalmaskF = np.random.randint(0, 2, (n_bands, 100)).astype(np.int8)
# traindatasets_maskF = np.random.randint(0, 2, (n_bands, 100)).astype(np.int8)
# traindatasets_deltaF = np.random.rand(n_bands, feature_num, 100).astype(np.float16)
# traindatasets_deltaBF = np.random.rand(n_bands, feature_num, 100).astype(np.float16)








# Lưu dữ liệu ngẫu nhiên dưới dạng JSON
fs = open('/mnt/data1tb/BRIOS/dataTrain/random_training_data.json', 'w')
all_len = traindatasets_valuesF.shape[2]
print('Save random training dataset as JSON: ')
for id_ in tqdm(range(all_len)):
    parse_idTrain(id_)
fs.close()