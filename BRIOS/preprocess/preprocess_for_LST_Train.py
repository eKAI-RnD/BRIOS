import numpy as np
from scipy.ndimage import generic_filter
from tqdm import tqdm
import os
import rasterio
from datetime import datetime, timedelta
import json

"""
Taoj duwx lieuej 
"""




def create_json_data(dir):
    def find_missing_date(list_lst, list_ndvi):
        date_lst = [f for f in list_lst if f.endswith('.tif')]
        date_lst = sorted(
            datetime.strptime(f.split('_')[1].split('.')[0], "%Y-%m-%d") for f in date_lst
        )
    
        date_ndvi = [f.split('.')[0].split('_')[-1] for f in list_ndvi]
    
        date_ndvi = sorted(
            datetime.strptime(f, "%Y-%m-%d") for f in date_ndvi
        )
        
        # Chuyển danh sách về dạng set để so sánh
        set_ndvi = set(date_ndvi)
        set_lst = set(date_lst)
        #print("LST:", set_lst)
        #print("NDVI", set_ndvi)

        # Tìm các ngày bị thiếu trong NDVI nhưng có trong RVI
        missing_dates = set_ndvi - set_lst
        # print(missing_dates)
        return list(missing_dates), date_ndvi
            
    def create_lst_time_series(folder_path, output_path, missing_dates, date_ndvi):
        # Dates with no data
        missing_dates = set(missing_dates)  # Convert to set for quick lookups

        # Check raster dimensions from a sample file
        with rasterio.open(os.path.join(folder_path, os.listdir(folder_path)[0])) as src:
            height, width = src.shape

        # Initialize list to hold time series data
        time_series = []

        # Generate the complete list of dates, assuming an 8-day interval
        start_date = date_ndvi[0]
        end_date = date_ndvi[-1]
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            file_path = os.path.join(folder_path, f"lst_{date_str}.tif") # TODO
            
            if date_str in missing_dates or not os.path.exists(file_path):
                # Append a null array for missing dates
                null_array = np.full((height, width), np.nan)
                time_series.append(null_array)
            else:
                # Read and store data for available dates
                with rasterio.open(file_path) as src:
                    time_series.append(src.read(1))  # Reads the first band

            # Increment by 8 days
            current_date += timedelta(days=16)

        # Stack along the time dimension, then transpose to (x, y, time)
        lst_data = np.stack(time_series, axis=0).transpose(1, 2, 0)

        np.save(output_path, lst_data)

        print(f"Data saved to {output_path}")

    
    def create_ndvi_time_series(folder_path, output_ndvi_path):
        # List all available .tif files in the folder
        available_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])

        ndvi_time_series = []

        for file_name in available_files:
            file_path = os.path.join(folder_path, file_name)
            with rasterio.open(file_path) as src:
                ndvi_time_series.append(src.read(1))  

        # Stack along the time dimension and transpose to (x, y, time)
        ndvi_data = np.stack(ndvi_time_series, axis=0).transpose(1, 2, 0)

        # Save each time series as .npy file
        np.save(output_ndvi_path, ndvi_data)

        print(f"RVI data saved to {output_ndvi_path}")
    
    def choose_data_validate(arr_index, valid_ratio = 0.1):
        num_valid = max(1, int(valid_ratio * len(arr_index)))
        index_for_valid = np.random.choice(arr_index, num_valid, replace=False)
        return index_for_valid

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

    def parse_rec(values, masks, eval_masks, deltas):
        rec = []
        for i in range(values.shape[0]):
            recone = {}
            recone['deltas'] = deltas[i, :].tolist()
            recone['masks'] = masks[i].astype('int8').tolist()
            recone['values'] = values[i, :].astype('float32').tolist()
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

    




    """
    Step create time series data
    ndvi_timeseries.shape = (x, y, t)
    """
    
    list_region = os.listdir(dir)
    ndvi_stack = []
    lst_stack = []
    for region in list_region:
        # child_region = (os.listdir(dir + region)[0]).split('_')[0]
        
        ndvi_raster_path = dir + region + f'/ndvi'
        lst_raster_path = dir + region + f'/lst'
        ndvi_time_series_path = dir + region + '/ndvi_timeseries.npy'
        lst_time_series_path = dir + region + '/lst_timeseries.npy'
        
        missing_date, date_lst= find_missing_date(os.listdir(lst_raster_path), os.listdir(ndvi_raster_path))
        #print(missing_date)
        create_lst_time_series(folder_path=lst_raster_path, output_path=lst_time_series_path, missing_dates=missing_date, date_ndvi=date_lst)
        create_ndvi_time_series(folder_path=ndvi_raster_path, output_ndvi_path=ndvi_time_series_path)

        

    for region in tqdm(list_region, desc="load region: "):
        ndvi_time_series_path = dir + region + '/ndvi_timeseries.npy'
        lst_time_series_path = dir + region + '/lst_timeseries.npy'
        ndvi_data_ = np.load(ndvi_time_series_path)
        lst_data_ = np.load(lst_time_series_path)
        print(f'ndvi_data shape: {ndvi_data_.shape}')
        print(f'lst data shape: {lst_data_.shape}')
        ndvi_data_ = ndvi_data_.reshape((ndvi_data_.shape[0] * ndvi_data_.shape[1], ndvi_data_.shape[2]))
        lst_data_ = lst_data_.reshape((lst_data_.shape[0] * lst_data_.shape[1], lst_data_.shape[2]))
        #print(ndvi_data_.shape)
        ndvi_stack.append(ndvi_data_)
        lst_stack.append(lst_data_)
        
    ndvi_data = np.concatenate(ndvi_stack, axis=0)  # Kết hợp theo chiều 0 (dọc)
    lst_data = np.concatenate(lst_stack, axis=0)    # Kết hợp theo chiều 0 (dọc)
    #print(ndvi_data.shape)
    # print(lst_data.shape)
    del ndvi_stack, lst_stack
    
    cloudMask = np.zeros(lst_data.shape)

    for series_index in range(lst_data.shape[0]):
        for time_step in range(lst_data.shape[1]):
            if np.isnan(lst_data[series_index, time_step]):
                cloudMask[series_index, time_step] = 1

    """
    Make train and valid data
    """
    data_num = np.full(lst_data.shape[0], lst_data.shape[1])
   
    for series_index in range(cloudMask.shape[0]):
        num_ones = len(np.argwhere(cloudMask[series_index] == 1))
        data_num[series_index] -= num_ones
    
    print('numdata: ', len(data_num>=12))
    # mask sar, 0: sar get nan, 1: fully data
    unvalid_ndvi = np.where(np.isnan(lst_data).any(axis=1), 1, 0)
    print(unvalid_ndvi)
    combined_mask = np.where((unvalid_ndvi == 1), 1, 0)
    index_for_train = np.where((data_num >= 12))[0]
    cloudMask0 = cloudMask

    for series_index in index_for_train:
        uncloud_index = np.where(cloudMask0[series_index] == 0)[0]
        fakecloud_index = choose_data_validate(uncloud_index)
        cloudMask0[series_index, fakecloud_index] = 2


    ndvi_input = ndvi_data[index_for_train]
    lst_input = lst_data[index_for_train]
    cloudmask_input = cloudMask0[index_for_train]

    print(f"ndvi_input: {ndvi_data}")
    print(f"ndvi first: {ndvi_data.shape}")
    print(f"index for train: {len(index_for_train)}")
    print(f"ndvi input shape: {ndvi_input.shape}")
    print(f"lst input shape: {lst_input.shape}")
    """
    have dataset contains: 
    - input data: [rvi, vh, ndvi] (n_series, n_features, n_timesteps)
    - mask_train: 1 => training, 0 => no
    - mask_eval: 1 => validating, 0 => no
    """
    mask_train = np.where(cloudmask_input == 0, 1, 0)
    mask_eval = np.where(cloudmask_input == 2, 1, 0)
    # input_data = np.stack([rvi_input, vh_input, ndvi_input], axis=1)

    feature_num = 2
    n_timesteps = 23
    time = np.arange(1,369,16) # timestep: 8

    deltaT_forward = np.zeros((len(index_for_train), n_timesteps))
    for i in range(len(index_for_train)):
        maskone = mask_train[i, :]
        done = cal_timestep(time, maskone)
        deltaT_forward[i, :] = done

    deltaT_backward = np.zeros((len(index_for_train), n_timesteps))
    for i in range(len(index_for_train)):
        maskone = mask_train[i, :]
        maskone = maskone[::-1]
        done = cal_timestep(time, maskone)
        deltaT_backward[i, :] = done

    print('Generate time interval for SAR data: ')
    deltaTt = np.zeros((len(index_for_train), n_timesteps))
    for i in range(len(index_for_train)):
        maskone = np.ones(n_timesteps)
        maskone = np.int_(maskone)
        done = cal_timestep(time, maskone)
        deltaTt[i, :] = done


    """
    Final create dataset
    """
    traindatasets_valuesF = np.empty((n_timesteps, feature_num, 0),dtype=np.float16)
    traindatasets_evalmaskF = np.empty((n_timesteps, 0),dtype=np.int8)
    traindatasets_maskF = np.empty((n_timesteps, 0),dtype=np.int8)
    traindatasets_deltaF = np.empty((n_timesteps, feature_num, 0),dtype=np.float16)
    traindatasets_deltaBF = np.empty((n_timesteps, feature_num, 0),dtype=np.float16)


    print('Generate training datasets: ')
    trainingdatasets_values = np.zeros((n_timesteps, feature_num, len(index_for_train)))
    for step in range(n_timesteps):
        for series_index in range(len(index_for_train)):
            trainingdatasets_values[step, 0, series_index] = ndvi_input[series_index, step]
            if np.isnan(lst_input[series_index, step]):
                trainingdatasets_values[step, 1, series_index] = -100
            else:
                trainingdatasets_values[step, 1, series_index] = lst_input[series_index, step]
    traindatasets_valuesF = np.concatenate((traindatasets_valuesF, trainingdatasets_values), axis=2)

    print('Generate evalmask: ')
    traindatasets_evalmask = np.zeros((n_timesteps,len(index_for_train)), dtype=np.int8)
    for step in range(n_timesteps):
        for series_index in range(len(index_for_train)):
            traindatasets_evalmask[step, series_index] = mask_eval[series_index, step]
    traindatasets_evalmaskF = np.concatenate((traindatasets_evalmaskF, traindatasets_evalmask), axis=1)

    print('Generate mask train: ')
    traindatasets_mask = np.zeros((n_timesteps, len(index_for_train)), dtype=np.int8)
    for step in range(n_timesteps):
        for series_index in range(len(index_for_train)):
            traindatasets_mask[step, series_index] = mask_train[series_index, step]
    traindatasets_maskF = np.concatenate((traindatasets_maskF, traindatasets_mask), axis=1)

    print('Generate training delta datasets: ')
    traindatasets_delta = np.zeros((n_timesteps, feature_num, len(index_for_train)), dtype=np.float16)
    for step in range(n_timesteps):
        for series_index in range(len(index_for_train)):
            traindatasets_delta[step, 0, series_index] = deltaTt[series_index, step]
            traindatasets_delta[step, 1, series_index] = deltaT_forward[series_index, step]
    traindatasets_deltaF = np.concatenate((traindatasets_deltaF, traindatasets_delta), axis=2)

    print('Generate training delta dataset backward: ')
    traindatasets_delta_backward = np.zeros((n_timesteps, feature_num, len(index_for_train)), dtype=np.float16)
    for step in range(n_timesteps):
        for series_index in range(len(index_for_train)):
            traindatasets_delta_backward[step, 0, series_index] = deltaTt[series_index, step]
            traindatasets_delta_backward[step, 1, series_index] = deltaT_backward[series_index, step]
    traindatasets_deltaBF = np.concatenate((traindatasets_deltaBF, traindatasets_delta_backward), axis=2)


    fs = open(dir + 'training_data.json', 'w')
    all_len = traindatasets_valuesF.shape[2]
    print('Save training dataset as JSON: ')
    for id_ in tqdm(range(all_len)):
        parse_idTrain(id_)
    fs.close()

create_json_data(dir='/mnt/storage/data/EOV_LST/Train_LST/Data_train_2/')


