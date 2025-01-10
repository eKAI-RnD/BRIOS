import numpy as np
from scipy.ndimage import generic_filter
from tqdm import tqdm
import os
import rasterio
from datetime import datetime, timedelta
import json

"""
Step 1: fill nan value in sar data
Step 2: fill mising data in ndvi data by nan value
Step 3: From raster, create time series data (x, y, t)
Step 4: Create json data include forward backward, with fields: values, deltas, masks, eval_masks
"""






def fillNA(rvi_data_path, vh_data_path, rvi_full_path, vh_full_path):
    """
    Fill missing values (NaN) in RVI and VH datasets and save the results.

    This function processes two datasets (RVI and VH) stored in `.npy` format. 
    It uses a custom nanmean filter to fill missing values in a 3D array (x, y, t), 
    applying the filter over a 7x7 spatial window for each time slice.
    The filled datasets are saved to the specified output paths.

    Parameters:
        rvi_data_path (str): Path to the input RVI dataset in `.npy` format.
        vh_data_path (str): Path to the input VH dataset in `.npy` format.
        rvi_full_path (str): Path to save the filled RVI dataset in `.npy` format.
        vh_full_path (str): Path to save the filled VH dataset in `.npy` format.

    Processing Steps:
        1. Load the RVI and VH datasets from the specified paths.
        2. For the RVI dataset:
           - Copy the data.
           - Apply the nanmean filter to each time slice to fill NaN values.
        3. For the VH dataset:
           - Copy the data.
           - Apply the nanmean filter to each time slice to fill NaN values.
        4. Count and print the number of remaining NaN values after filling.
        5. Save the processed datasets to the specified output paths.

    Output:
        - Prints the number of NaN values remaining in the VH dataset after processing.
        - Saves the filled RVI and VH datasets to the specified output paths.
    """

    def nanmean_filter(values):
        valid_values = values[~np.isnan(values)]
        return np.mean(valid_values) if valid_values.size > 0 else np.nan

    rvi_data = np.load(rvi_data_path)
    vh_data = np.load(vh_data_path)

    cnt = 0
    for x in range(vh_data.shape[0]):
        for y in range(vh_data.shape[1]):
            for t in range(vh_data.shape[2]):
                if np.isnan(vh_data[x, y, t]):
                    cnt += 1
                
    print(f'truoc khi fill co {cnt} gia tri nan')

    data = rvi_data[...,1]

    filled_data = data.copy()
    nan_positions = np.isnan(np.array(filled_data))
    filled_data[nan_positions] = generic_filter(filled_data, nanmean_filter, size=7)[nan_positions]

    rvi_filled_data = rvi_data.copy()
    for i in tqdm(range(rvi_data.shape[2]), desc='filling rvi dataset:   '):
        nan_pos = np.isnan(rvi_filled_data[...,i])
        rvi_filled_data[nan_pos,i] = generic_filter(rvi_filled_data[...,i], nanmean_filter, size=7)[nan_pos]

    vh_filled_data = vh_data.copy()
    for i in tqdm(range(vh_data.shape[2]), desc='filling vh dataset:   '):
        nan_pos_vh = np.isnan(vh_filled_data[...,i])
        vh_filled_data[nan_pos_vh,i] = generic_filter(vh_filled_data[...,i], nanmean_filter, size=7)[nan_pos_vh]

    cnt = 0
    for x in range(vh_filled_data.shape[0]):
        for y in range(vh_filled_data.shape[1]):
            for t in range(vh_filled_data.shape[2]):
                if np.isnan(vh_filled_data[x, y, t]):
                    cnt += 1
                
    print(f'sau khi fill con {cnt} gia tri nan')

    np.save(rvi_full_path, rvi_filled_data)
    np.save(vh_full_path, vh_filled_data)








def create_json_data(dir):
    
    def find_missing_date(list_ndvi, list_rvi):
        dates_ndvi = [f for f in list_ndvi if f.endswith('.tif')]
        dates_ndvi = sorted(
            datetime.strptime(f.split('_')[1].split('.')[0], "%Y-%m-%d") for f in dates_ndvi
        )
    
        dates_rvi = [f.split('.')[0].split('_')[-1] for f in list_rvi]
    
        dates_rvi = sorted(
            datetime.strptime(f, "%Y-%m-%d") for f in dates_rvi
        )
        
        # Chuyển danh sách về dạng set để so sánh
        set_rvi = set(dates_rvi)
        set_ndvi = set(dates_ndvi)

        # Tìm các ngày bị thiếu trong NDVI nhưng có trong RVI
        missing_dates = set_rvi - set_ndvi
        print(missing_dates)
        return list(missing_dates), dates_rvi
            
    def create_ndvi_time_series(folder_path, output_path, missing_dates, date_rvi):
        """
        Create a time-series NDVI dataset from a folder of raster files.

        This function reads NDVI raster files from a specified folder, handles missing dates by 
        adding arrays filled with NaN values, and saves the final time-series dataset as a 
        NumPy array. The time-series data is saved with the shape (height, width, time).

        Parameters:
            folder_path (str): Path to the folder containing NDVI `.tif` files.
                - Files should be named in the format `ndvi8days_YYYY-MM-DD.tif`.
            output_path (str): Path to save the resulting `.npy` file.

        Processing Details:
            1. Identifies missing dates from a predefined list and assigns a NaN-filled array for those dates.
            2. Reads NDVI data from available `.tif` files, using the first band of each file.
            3. Assumes an 8-day interval between data points and generates a complete time-series.
            4. Combines all data into a single NumPy array with the shape `(height, width, time)`.

        Outputs:
            - A `.npy` file containing the time-series NDVI data.
            - Prints a message confirming the save location.

        Notes:
            - If a date is missing or a corresponding `.tif` file is not found, the function fills that
            time step with an array of NaN values.
            - The dimensions of the output data are based on the first `.tif` file in the folder.

        Example:
            create_ndvi_time_series(
                folder_path="/path/to/ndvi_rasters",
                output_path="/path/to/output/ndvi_timeseries.npy"
            )
        """

        # Dates with no data
        missing_dates = set(missing_dates)  # Convert to set for quick lookups

        # Check raster dimensions from a sample file
        with rasterio.open(os.path.join(folder_path, os.listdir(folder_path)[0])) as src:
            height, width = src.shape

        # Initialize list to hold time series data
        time_series = []

        # Generate the complete list of dates, assuming an 8-day interval
        start_date = date_rvi[0]
        end_date = date_rvi[-1]
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            file_path = os.path.join(folder_path, f"ndvi8days_{date_str}.tif")
            
            if date_str in missing_dates or not os.path.exists(file_path):
                # Append a null array for missing dates
                null_array = np.full((height, width), np.nan)
                time_series.append(null_array)
            else:
                # Read and store data for available dates
                with rasterio.open(file_path) as src:
                    time_series.append(src.read(1))  # Reads the first band

            # Increment by 8 days
            current_date += timedelta(days=8)

        # Stack along the time dimension, then transpose to (x, y, time)
        ndvi_data = np.stack(time_series, axis=0).transpose(1, 2, 0)

        np.save(output_path, ndvi_data)

        print(f"Data saved to {output_path}")

    
    def create_sar_time_series(folder_path, output_rvi_path, output_vh_path):
        """
        Create time-series datasets for RVI and VH from SAR raster files.

        This function reads dual-band SAR raster files from a specified folder, extracts 
        RVI data from the first band and VH data from the second band, and saves each as 
        a separate time-series `.npy` file. The resulting arrays are shaped `(height, width, time)`.

        Parameters:
            folder_path (str): Path to the folder containing SAR `.tif` files.
                - Files should be named consistently to ensure correct temporal ordering.
            output_rvi_path (str): Path to save the resulting RVI time-series `.npy` file.
            output_vh_path (str): Path to save the resulting VH time-series `.npy` file.

        Processing Details:
            1. Lists all `.tif` files in the folder and sorts them for temporal consistency.
            2. Extracts RVI data (band 1) and VH data (band 2) from each raster file.
            3. Stacks the extracted data along the time dimension.
            4. Saves the resulting RVI and VH time-series as separate `.npy` files.

        Outputs:
            - Two `.npy` files containing the RVI and VH time-series data, respectively.
            - Prints the shapes of the resulting arrays and the paths where they are saved.

        Notes:
            - Assumes that all `.tif` files in the folder have the same spatial dimensions and band structure.
            - Temporal ordering of files is based on the alphabetical order of filenames.

        Example:
            create_sar_time_series(
                folder_path="/path/to/sar_rasters",
                output_rvi_path="/path/to/output/rvi_timeseries.npy",
                output_vh_path="/path/to/output/vh_timeseries.npy"
            )
        """
        # List all available .tif files in the folder
        available_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])

        # Check raster dimensions from a sample file
        with rasterio.open(os.path.join(folder_path, available_files[0])) as src:
            height, width = src.shape

        # Initialize lists to hold RVI and VH time series data
        rvi_time_series = []
        vh_time_series = []

        # Read and store RVI and VH data from each file
        for file_name in available_files:
            file_path = os.path.join(folder_path, file_name)
            with rasterio.open(file_path) as src:
                # Append the first band for RVI data
                rvi_time_series.append(src.read(1))  # RVI is in band 1
                # Append the second band for VH data
                vh_time_series.append(src.read(2))  # VH is in band 2

        # Stack along the time dimension and transpose to (x, y, time)
        rvi_data = np.stack(rvi_time_series, axis=0).transpose(1, 2, 0)
        vh_data = np.stack(vh_time_series, axis=0).transpose(1, 2, 0)

        # Save each time series as .npy file
        np.save(output_rvi_path, rvi_data)
        np.save(output_vh_path, vh_data)

        print(f"RVI data saved to {output_rvi_path}")
        print(f"VH data saved to {output_vh_path}")
    
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
    region = 'ThuyVan-ThaiThuy-ThaiBinh/'
    ndvi_raster_path = dir + region + 'thuyvan_ndvi8days'
    sar_raster_path = dir + region + 'thuyvan_rvi_8days'
    ndvi_time_series_path = dir + region + 'ndvi_timeseries.npy'
    rvi_time_series_path = dir + region + 'rvi_timeseries.npy'
    vh_time_series_path = dir + region + 'vh_timeseries.npy'
    missing_date, date_rvi = find_missing_date(os.listdir(ndvi_raster_path), os.listdir(sar_raster_path))
    create_ndvi_time_series(folder_path=ndvi_raster_path, output_path=ndvi_time_series_path, missing_dates=missing_date, date_rvi=date_rvi)
    create_sar_time_series(folder_path=sar_raster_path, output_rvi_path=rvi_time_series_path, output_vh_path=vh_time_series_path)

    fillNA(rvi_data_path=rvi_time_series_path, vh_data_path=vh_time_series_path,
       rvi_full_path=rvi_time_series_path, vh_full_path=vh_time_series_path)


    ndvi_data = np.load(ndvi_time_series_path)
    rvi_data = np.load(rvi_time_series_path)
    vh_data = np.load(vh_time_series_path)

    """
    save coordinate x y t
    """
    x, y, t = ndvi_data.shape
    coordinates = np.array([(i, j) for i in range(x) for j in range(y)])
    np.save(dir + region + 'coordinates.npy', coordinates)


    ndvi_data = ndvi_data.reshape((ndvi_data.shape[0] * ndvi_data.shape[1], ndvi_data.shape[2]))
    rvi_data = rvi_data.reshape((rvi_data.shape[0] * rvi_data.shape[1], rvi_data.shape[2]))
    vh_data = vh_data.reshape((vh_data.shape[0] * vh_data.shape[1], vh_data.shape[2]))



    """
    Step make cloudmask
    - `ndvi_data[i]` == `np.nan` => have cloud
    - `cloudMask[i] = 0`: no cloud 
    - `cloudMask[i]` = 1: cloud
    """
    cloudMask = np.zeros(ndvi_data.shape)

    for series_index in range(ndvi_data.shape[0]):
        for time_step in range(ndvi_data.shape[1]):
            if np.isnan(ndvi_data[series_index, time_step]):
                cloudMask[series_index, time_step] = 1

    """
    Make train and valid data
    """
    data_num = np.full(ndvi_data.shape[0], ndvi_data.shape[1])
    # for series_index in range(cloudMask.shape[0]):
    #     num_ones = len(np.argwhere(cloudMask[series_index] == 1))
    #     data_num[series_index] -= num_ones
    
    # mask sar, 0: sar get nan, 1: fully data
    unvalid_rvi = np.where(np.isnan(rvi_data).any(axis=1), 1, 0)
    unvalid_vh = np.where(np.isnan(vh_data).any(axis=1), 1, 0)
    combined_mask = np.where(np.logical_or(unvalid_rvi == 1, unvalid_vh == 1), 1, 0)

    index_for_train = [i for i in range(ndvi_data.shape[0])]
    cloudMask0 = cloudMask

    # for series_index in index_for_train:
    #     uncloud_index = np.where(cloudMask0[series_index] == 0)[0]
    #     fakecloud_index = choose_data_validate(uncloud_index)
    #     cloudMask0[series_index, fakecloud_index] = 2


    ndvi_input = ndvi_data[index_for_train]
    rvi_input = rvi_data[index_for_train]
    vh_input = vh_data[index_for_train]
    cloudmask_input = cloudMask0[index_for_train]

    print(f"ndvi_input: {ndvi_data}")
    print(f"ndvi first: {ndvi_data.shape}")
    print(f"index for train: {len(index_for_train)}")
    print(f"ndvi input shape: {ndvi_input.shape}")
    print(f"rvi input shape: {rvi_input.shape}")
    print(f"vh input shape: {vh_input.shape}")

    """
    have dataset contains: 
    - input data: [rvi, vh, ndvi] (n_series, n_features, n_timesteps)
    - mask_train: 1 => training, 0 => no
    - mask_eval: 1 => validating, 0 => no
    """
    mask_train = np.where(cloudmask_input == 0, 1, 0)
    mask_eval = np.where(cloudmask_input == 2, 1, 0)
    # input_data = np.stack([rvi_input, vh_input, ndvi_input], axis=1)

    feature_num = 3
    n_timesteps = 46
    time = np.arange(1,369,8) # timestep: 8

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

    # ndvi_input_backward = ndvi_input[:, ::-1]
    # rvi_input_backward = rvi_input[:, ::-1]
    # vh_input_backward = vh_input[:, ::-1]

    # input_data_backward = np.stack([rvi_input_backward, vh_input_backward, ndvi_input_backward], axis=1)


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
            trainingdatasets_values[step, 0, series_index] = vh_input[series_index, step]
            trainingdatasets_values[step, 1, series_index] = rvi_input[series_index, step]
            if np.isnan(ndvi_input[series_index, step]):
                trainingdatasets_values[step, 2, series_index] = -100
            else:
                trainingdatasets_values[step, 2, series_index] = ndvi_input[series_index, step]
    traindatasets_valuesF = np.concatenate((traindatasets_valuesF, trainingdatasets_values), axis=2)

    print('Generate evalmask: ')
    traindatasets_evalmask = np.zeros((n_timesteps,len(index_for_train)), dtype=np.int8)
    for step in range(n_timesteps):
        for series_index in range(len(index_for_train)):
            traindatasets_evalmask[step, series_index] = mask_eval[series_index, step]
    traindatasets_evalmaskF = np.concatenate((traindatasets_evalmaskF, traindatasets_evalmask), axis=1)

    print('Generate mask tran: ')
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
            traindatasets_delta[step, 1, series_index] = deltaTt[series_index, step]
            traindatasets_delta[step, 2, series_index] = deltaT_forward[series_index, step]
    traindatasets_deltaF = np.concatenate((traindatasets_deltaF, traindatasets_delta), axis=2)

    print('Generate training delta dataset backward: ')
    traindatasets_delta_backward = np.zeros((n_timesteps, feature_num, len(index_for_train)), dtype=np.float16)
    for step in range(n_timesteps):
        for series_index in range(len(index_for_train)):
            traindatasets_delta_backward[step, 0, series_index] = deltaTt[series_index, step]
            traindatasets_delta_backward[step, 1, series_index] = deltaTt[series_index, step]
            traindatasets_delta_backward[step, 2, series_index] = deltaT_backward[series_index, step]
    traindatasets_deltaBF = np.concatenate((traindatasets_deltaBF, traindatasets_delta_backward), axis=2)


    fs = open(dir + region + 'training_data.json', 'w')
    all_len = traindatasets_valuesF.shape[2]
    print('Save training dataset as JSON: ')
    for id_ in tqdm(range(all_len)):
        parse_idTrain(id_)
    fs.close()






# fillNA(rvi_data_path='/mnt/data1tb/brios/BRIOS/datasets/data2/rvi_timeseries.npy', vh_data_path='/mnt/data1tb/brios/BRIOS/datasets/data2/vh_timeseries.npy',
#        rvi_full_path='/mnt/data1tb/brios/BRIOS/datasets/dataTrain/rvi_full.npy', vh_full_path='/mnt/data1tb/brios/BRIOS/datasets/dataTrain/vh_full.npy')

create_json_data(dir='/mnt/storage/code/EOV_NDVI/brios/datasets/Train_LST/')