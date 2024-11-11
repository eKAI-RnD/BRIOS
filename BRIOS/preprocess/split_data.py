import numpy as np
import json
from tqdm import tqdm


def savePreprocessedData(path, data):
    """ Save preprocessed Data to npy array

    Args:
        path (string): path to shape data
        data (array): array input
    """
    with open(path +".npy", 'bw') as outfile:
        np.save(outfile, data)




# parameters
n_bands = 46
n_samples = 160740


# cloudmask
cloudmask = np.load('/mnt/storage/huyekgis/brios/data2/numpy_data/cloudmask.npy')
areamask = np.load('/mnt/storage/huyekgis/brios/data2/numpy_data/areamask.npy')

# print(f"areamask: {areamask}")

# datanum
datanum = np.full(n_samples, n_bands, dtype=np.int8)
for i in range(n_samples):
    datanum[i] -= np.sum(cloudmask[i] == 1)

print(f"DATANUM after adjustment: {datanum}")

# idx
idx = np.argwhere((areamask != 0) & (datanum > 25)).flatten()
train_size = int(len(idx)* 1.0) 

train_index = np.random.choice(idx, size=train_size, replace=False)
# test_index = np.setdiff1d(idx, train_index)

print(f"TRAIN INDEX: {train_index.shape}")
# print(f"TEST INDEX: {test_index.shape}")

dir = "/mnt/storage/huyekgis/brios/dataTrain/"
savePreprocessedData(dir + "train_index", train_index)
# savePreprocessedData(dir + "test_index", test_index)
savePreprocessedData(dir + "cloudmask", cloudmask)