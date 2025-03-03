import torch
import torch.optim as optim
from scipy.stats import pearsonr
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import numpy as np
import utils
import models
from support.early_stopping import EarlyStopping
import batch_data_loader
from math import sqrt
from sklearn import metrics
from tqdm import tqdm
import json
import os
from dotenv import load_dotenv
import time
import mlflow
import mlflow.pytorch
import logging as logging1

load_dotenv()

# Load config
ROOT_PATH_PRJ = os.getenv('ROOT_PATH')
MODEL_NAME = os.getenv('MODEL_NAME')
MODEL_SAVE = os.getenv('MODEL_SAVE')
DATA_TRAIN_PATH = os.getenv('DATA_TRAIN')
METRIC_SAVE = os.getenv('METRIC_SAVE')
PRE_TRAIN = os.getenv('PRE_TRAIN')
PRE_TRAIN_PATH = os.getenv('PRE_TRAIN_PATH')
PRE_EPOCHS = os.getenv("PRE_TRAIN_EPOCHS")  # Pre-trained epochs as a string

# Parameter setting
epochs = 100
batch_size = 1024
hid_size = 96

# Train NDVI
SEQ_LEN = 46
INPUT_SIZE = 3
SELECT_SIZE = 1

def trainModel():
    print('---------------------')
    print(f'Training model {MODEL_NAME}')
    print('---------------------')
    
    # Instantiate model from models module
    model = getattr(models, MODEL_NAME).Model(hid_size, INPUT_SIZE, SEQ_LEN, SELECT_SIZE)
    
    # Determine starting epoch (default 0) and load pretrained weights if available
    start_epoch = 0
    if PRE_TRAIN and PRE_TRAIN.lower() == "true" and os.path.exists(PRE_TRAIN_PATH):
        state_dict = torch.load(PRE_TRAIN_PATH)
        model.load_state_dict(state_dict)
        print(f"Loaded pre-trained weights from {PRE_TRAIN_PATH}")
        try:
            start_epoch = int(PRE_EPOCHS)
            print(f"Starting from pre-trained epoch: {start_epoch}")
        except Exception as e:
            print("Error converting PRE_TRAIN_EPOCHS to int, starting from 0")
            start_epoch = 0
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total params is {}'.format(total_params))
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Early Stopping setup
    SavePath = MODEL_SAVE  # Model parameter save path
    patience = 20
    early_stopping = EarlyStopping(savepath=SavePath, patience=patience, verbose=True, useralystop=False, delta=0.001)
    
    # Optimizer and learning rate scheduler
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.0008)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    
    # Load training data
    print("Loading data loader...")
    train_path = DATA_TRAIN_PATH
    start_time = time.time()
    data_iter = batch_data_loader.get_train_loader(batch_size=batch_size, prepath=train_path)
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Data loader execution time: {execution_time:.2f} seconds")
    
    # Lists to store metrics for each epoch
    train_losses = []
    rmses = []
    
    # MLflow Experiment
    with mlflow.start_run() as run:
        # Log experiment parameters
        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "hid_size": hid_size,
            "SEQ_LEN": SEQ_LEN,
            "INPUT_SIZE": INPUT_SIZE,
            "SELECT_SIZE": SELECT_SIZE,
            "lr": 1e-3,
            "weight_decay": 0.0008,
            "start_epoch": start_epoch
        })
        
        print("Start training")
        # Use current_epoch = epoch + start_epoch so that logging starts from the pre-trained epoch count
        for epoch in range(epochs):
            current_epoch = epoch + start_epoch
            model.train()
            run_loss = 0.0
            rmse = 0.0
            validnum = 0.0
            for idx, data in tqdm(enumerate(data_iter), total=len(data_iter), 
                                    desc=f'EPOCH[{current_epoch+1}/{epochs + start_epoch}]'):
                data = utils.to_var(data)
                ret = model.run_on_batch(data, optimizer, current_epoch)
                run_loss += ret['loss'].item()
                
                eval_masks = ret['eval_masks'].data.cpu().numpy()
                count_ones = np.count_nonzero(eval_masks == 1)
                
                if count_ones != 0:
                    series_f = ret['imputations_f'].data.cpu().numpy()
                    series_b = ret['imputations_b'].data.cpu().numpy()
                    series_f = series_f[np.where(eval_masks == 1)]
                    series_b = series_b[np.where(eval_masks == 1)]
                    validnum += 1.0
                
                if (current_epoch + 1) % 20 == 0 and count_ones != 0:
                    eval_ = ret['evals'].data.cpu().numpy()
                    imputation = ret['imputations'].data.cpu().numpy()
                    eval_ = eval_[np.where(eval_masks == 1)]
                    imputation = imputation[np.where(eval_masks == 1)]
                    rmse += sqrt(metrics.mean_squared_error(eval_, imputation))
            
            # Calculate epoch metrics
            train_loss = run_loss / (idx + 1.0)
            train_losses.append(train_loss)
            print(f"Epoch {current_epoch + 1}: Loss: {train_loss}")
            mlflow.log_metric('train_loss', train_loss, step=current_epoch + 1)
            
            if (current_epoch + 1) % 20 == 0:
                validation_rmse = rmse / (validnum + 1e-5)
                rmses.append(validation_rmse)
                print(f"Epoch {current_epoch + 1}: Validation RMSE: {validation_rmse}")
                mlflow.log_metric('validation_rmse', validation_rmse, step=current_epoch + 1)
            
            # Save model checkpoint every 5 epochs using the adjusted epoch count
            if (current_epoch + 1) % 5 == 0:
                checkpoint_path = MODEL_SAVE
                torch.save(model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Early stopping check
            early_stopping(train_loss, model)
            if early_stopping.early_stop:
                print("Early stopping triggered")
                break
            
            scheduler.step(train_loss)
            
            # Save metrics to a JSON file every 20 epochs
            if (current_epoch + 1) % 20 == 0:
                metrics_history = {
                    'losses': train_losses,
                    'rmses': rmses
                }
                with open(f'{METRIC_SAVE}/lst_test_thong_qt.json', 'w') as f:
                    json.dump(metrics_history, f)
        
        # Log final model to MLflow
        mlflow.pytorch.log_model(model, "model")
        print("Model logged to MLflow")
    
    print("Training complete and metrics saved.")

if __name__ == '__main__':
    trainModel()
