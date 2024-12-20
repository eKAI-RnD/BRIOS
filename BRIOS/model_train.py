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
import json
from torch.optim.lr_scheduler import ReduceLROnPlateau
import mlflow
import mlflow.pytorch
import logging


def savePreprocessedData(path, data):
    with open(path + ".npy", 'bw') as outfile:
        np.save(outfile, data)

# parameter setting
epochs = 1000
batch_size = 1024
model_name = 'brios_attention'
hid_size = 96
SEQ_LEN = 46
INPUT_SIZE = 3
SELECT_SIZE = 1

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/mnt/storage/huyekgis/brios/BRIOS/models/logs/training_log.log'),  # Đường dẫn file log
        logging.StreamHandler()  # In ra console
    ]
)
logger = logging.getLogger(__name__)
logging.info('hihihi')
print('ffffff')
# Hàm trainModel
def trainModel():
    logger.info('---------------------')
    logger.info(f'Training model {model_name}')
    logger.info('---------------------')
    
    model = getattr(models, model_name).Model(hid_size, INPUT_SIZE, SEQ_LEN, SELECT_SIZE)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Total params is {}'.format(total_params))

    if torch.cuda.is_available():
        model = model.cuda()

    # Early Stopping setup
    SavePath = '/mnt/storage/huyekgis/brios/BRIOS/models/model_file/brios_attention/brios_attention_2.pt'
    patience = 20
    early_stopping = EarlyStopping(savepath=SavePath, patience=patience, verbose=True, useralystop=False, delta=0.001)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0008)

    # Scheduler: ReduceLROnPlateau
    #scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=10, verbose=True
    )

    
    # Load training data
    logger.info('Loading data loader...')
    train_path = '/mnt/storage/huyekgis/brios/datasets/dataTrain/training_data_1.json'
    data_iter = batch_data_loader.get_train_loader(batch_size=batch_size, prepath=train_path)

    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("hid_size", hid_size)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", 1e-3)

        # Lists to store metrics for each epoch
        train_losses = []
        valid_losses = []
        rmses = []

        # Setup TensorBoard writer
        logger.info('Start Training')

        for epoch in range(epochs):
            model.train()
            run_loss = 0.0
            rmse = 0.0
            validnum = 0.0

            for idx, data in tqdm(enumerate(data_iter), total=len(data_iter), desc=f'EPOCH[{epoch + 1}/{epochs}]'):
                data = utils.to_var(data)
                ret = model.run_on_batch(data, optimizer, epoch)
                run_loss += ret['loss'].item()

                eval_masks = ret['eval_masks'].data.cpu().numpy()
                count_ones = np.count_nonzero(eval_masks == 1)

                if count_ones != 0:
                    series_f = ret['imputations_f'].data.cpu().numpy()
                    series_b = ret['imputations_b'].data.cpu().numpy()
                    series_f = series_f[np.where(eval_masks == 1)]
                    series_b = series_b[np.where(eval_masks == 1)]
                    validnum += 1.0

                if (epoch + 1) % 10 == 0 and count_ones != 0:
                    eval_ = ret['evals'].data.cpu().numpy()
                    imputation = ret['imputations'].data.cpu().numpy()
                    eval_ = eval_[np.where(eval_masks == 1)]
                    imputation = imputation[np.where(eval_masks == 1)]
                    rmse += sqrt(metrics.mean_squared_error(eval_, imputation))

            # Calculate epoch metrics
            train_loss = run_loss / (idx + 1.0)
            train_losses.append(train_loss)

            logger.info(f"Epochs: {epoch + 1}, Loss: {train_loss}")

            # Log the training loss to TensorBoard and MLflow
            mlflow.log_metric("train_loss", train_loss, step=epoch + 1)

            if (epoch + 1) % 10 == 0:
                validation_rmse = rmse / (validnum + 1e-5)
                rmses.append(validation_rmse)
                logger.info(f"Epochs: {epoch + 1}, Validation AUC metrics: {validation_rmse}")
                
                # Log the validation RMSE to TensorBoard and MLflow
                mlflow.log_metric("validation_rmse", validation_rmse, step=epoch + 1)

            valid_losses.append(train_loss)

            # Early stopping
            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                logger.info("Early stopping")
                break

            scheduler.step(train_loss)

        # Save the trained model to MLflow
        mlflow.pytorch.log_model(model, "models")

        # Save metrics to a JSON file every 20 epochs
        if (epoch + 1) % 10 == 0:
            metrics_history = {
                'losses': train_losses,
                'rmses': rmses
            }
            with open('/mnt/storage/huyekgis/brios/BRIOS/models/history/history_wdsz2_attention.json', 'w') as f:
                json.dump(metrics_history, f)
        
        logger.info("Training complete and metrics saved.")

if __name__ == '__main__':
    trainModel()
