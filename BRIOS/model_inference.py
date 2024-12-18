import torch
import numpy as np
import utils
import models
import batch_data_loader
from tqdm import tqdm

# parameter setting
batch_size = 512
model_name = 'brios'
hid_size = 96
SEQ_LEN = 46
INPUT_SIZE = 3
SELECT_SIZE = 1

# predicting process
def ExecuteModel():

    print('=======')
    print('predicting')
    print('=======')
    model = getattr(models, model_name).Model(hid_size, INPUT_SIZE, SEQ_LEN, SELECT_SIZE)
    total_params = sum(p.numel() for p in model.parameters()
                       if p.requires_grad)
    print('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    SavePath = '/mnt/storage/huyekgis/brios/BRIOS/models/model_file/train_largedata/brios_base_retrain.pt'  #Model parameter path

    model.load_state_dict(torch.load(SavePath))

    # load input data

    data_path = '/mnt/storage/huyekgis/brios/RAW_TEST_Data/Data4BRIOS_TEST/AnPhus-Pleicu-GiaLai/training_data.json'
    data_iter = batch_data_loader.get_test_loader(batch_size=batch_size, prepath=data_path)

    model.eval()

    evals = []
    imputations = []

    save_impute = []

    for idx, data in tqdm(enumerate(data_iter), total=len(data_iter)):
        data = utils.to_var(data)
        ret = model.run_on_batch(data, None)

        eval_masks = ret['eval_masks'].data.cpu().numpy()
        eval_ = ret['evals'].data.cpu().numpy()
        imputation = ret['imputations'].data.cpu().numpy()
        masks = ret['masks'].data.cpu().numpy()

        imputation_fill = imputation
        save_impute.append(imputation_fill)
        del eval_, eval_masks, imputation, imputation_fill, masks

    evals = np.asarray(evals)
    imputations = np.asarray(imputations)

    del evals, imputations
    
    save_impute = np.concatenate(save_impute, axis=0)

    """
    convert time series to x y t
    first: save_impute have shape: like (183744, 46, 1)
    """
    save_impute.reshape(save_impute.shape[0], save_impute.shape[1])

    ndvi_root = np.load('/mnt/storage/huyekgis/brios/RAW_TEST_Data/Data4BRIOS_TEST/AnPhus-Pleicu-GiaLai/ndvi_timeseries.npy')
    x, y, t = ndvi_root.shape

    coordinates = np.load('/mnt/storage/huyekgis/brios/RAW_TEST_Data/Data4BRIOS_TEST/AnPhus-Pleicu-GiaLai/coordinates.npy')

    img1 = np.zeros((ndvi_root.shape[0], ndvi_root.shape[1], 46))  # Hoặc np.full((x_size, y_size, 46), np.nan)

    # Gán giá trị từ `save_impute` vào `img1` ở các tọa độ và thời điểm khác nhau
    for i in range(save_impute.shape[0]):
        for t in range(save_impute.shape[1]):  # Với `t` là chiều thời gian trong `save_impute`
            img1[coordinates[i, 0], coordinates[i, 1], t] = save_impute[i, t]

    
    print(f"result shape: {img1.shape}")



    resultpath = '/mnt/storage/huyekgis/brios/BRIOS/results/anphu/test_anphu.npy'   #predicted values save path

    np.save(resultpath, img1)

    del save_impute, data_iter, data, ret, model


if __name__ == '__main__':
    ExecuteModel()