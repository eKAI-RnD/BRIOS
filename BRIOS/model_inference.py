# import torch
# import numpy as np
# from math import sqrt
# from sklearn import metrics
# import utils
# import models
# import batch_data_loader
# from tqdm import tqdm
# # parameters
# model_name = 'brios'
# hid_size = 96
# SEQ_LEN = 46
# INPUT_SIZE = 3
# SELECT_SIZE = 1
# SavePath = '/mnt/storage/huyekgis/brios/BRIOS/models/test.pt'  # Model parameter load path


# def load_model():
#     model = getattr(models, model_name).Model(hid_size, INPUT_SIZE, SEQ_LEN, SELECT_SIZE)
#     model.load_state_dict(torch.load(SavePath))
#     model.eval()  # Set model to evaluation mode
#     if torch.cuda.is_available():
#         model = model.cuda()
#     return model


# def inference(input_data):
#     model = load_model()
#     all_predictions = [] 
#     with torch.no_grad():  
#         for idx, data in tqdm(enumerate(input_data), total=len(data_iter)):
#             data = utils.to_var(data)
#             ret = model.run_on_batch(data, None)
#             predictions = ret['imputations'].data.cpu().numpy() 
#             all_predictions.append(predictions)
#     # Gộp tất cả predictions lại thành một mảng numpy
#     all_predictions = np.concatenate(all_predictions, axis=0)
#     return all_predictions


# # Example usage of inference function
# if __name__ == '__main__':
#     data_path = '/mnt/storage/huyekgis/brios/dataTrain/anphu_kinhmon.json'
#     data_iter = batch_data_loader.get_test_loader(batch_size=1024, prepath=data_path)
#     predictions = inference(data_iter)
#     np.save('/mnt/storage/huyekgis/brios/BRIOS/models/predictions.npy', predictions)



import torch
import torch.optim as optim
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import utils
import models
from support.early_stopping import EarlyStopping
import batch_data_loader
from math import sqrt
from sklearn import metrics
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

    SavePath = '/mnt/storage/huyekgis/brios/BRIOS/models/test.pt'  #Model parameter path

    model.load_state_dict(torch.load(SavePath))

    # load input data
    data_path = '/mnt/storage/huyekgis/brios/dataTrain/anphu_kinhmon_75.json'
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

    resultpath = '/mnt/storage/huyekgis/brios/BRIOS/models/anphu_test_75.npy'   #predicted values save path
    np.save(resultpath, save_impute)

    del save_impute, data_iter, data, ret, model


if __name__ == '__main__':
    ExecuteModel()