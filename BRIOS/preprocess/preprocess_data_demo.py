import os
import rasterio
import numpy as np
from datetime import datetime, timedelta


def count_nans_and_cloudmask(ndvi_data, cloudmask):
    """
    Count the total number of NaNs in ndvi_data and the total number of 1s in cloudmask.

    Parameters:
    ndvi_data (numpy.ndarray): NDVI data with shape (x, y, time).
    cloudmask (numpy.ndarray): Cloud mask array with the same shape as ndvi_data.

    Returns:
    tuple: A tuple containing (total NaNs in ndvi_data, total 1s in cloudmask).
    """
    # Tổng số NaN trong ndvi_data
    total_nans = np.sum(np.isnan(ndvi_data))
    
    # Tổng số 1 trong cloudmask
    total_cloudmask_ones = np.sum(cloudmask == 1)
    
    return total_nans, total_cloudmask_ones


def check_nan_cloudmask(ndvi_data, cloudmask):
    """
    Check if cloudmask is 1 at positions where ndvi_data is NaN.

    Parameters:
    ndvi_data (numpy.ndarray): NDVI data with shape (x, y, time).
    cloudmask (numpy.ndarray): Cloud mask array with the same shape as ndvi_data.

    Returns:
    bool: True if cloudmask is 1 wherever ndvi_data is NaN, False otherwise.
    """
    # Tìm vị trí NaN trong ndvi_data
    nan_indices = np.isnan(ndvi_data)
    
    # Kiểm tra giá trị của cloudmask tại các vị trí NaN
    is_cloudmask_correct = np.all(cloudmask[nan_indices] == 1)
    
    print(f"================= {is_cloudmask_correct}")

    total_nans, total_cloudmask_ones = count_nans_and_cloudmask(ndvi_data, cloudmask)

    print(f"Tổng số NaN trong ndvi_data: {total_nans}")
    print(f"Tổng số giá trị 1 trong cloudmask: {total_cloudmask_ones}")



def process_cloudmask(ndvi_data, cloud_percentage=0.1):
    """
    Generate a cloud mask based on NDVI data and assign 10% of cloud mask `1` values to `2`.

    Parameters:
    ndvi_data (numpy.ndarray): NDVI data with shape (x, y, time), where missing values are NaN.
    cloud_percentage (float): Percentage of cloud mask `1` values to convert to `2`. Default is 0.1 (10%).

    Returns:
    numpy.ndarray: Cloud mask array with shape (x, y, time), where 1 indicates missing NDVI (cloud),
                   2 indicates a subset of cloud-affected data, and 0 indicates valid data.
    """
    cloudmask_data = np.zeros((ndvi_data.shape[0], ndvi_data.shape[1], ndvi_data.shape[2]))
    # Initial cloud mask: 1 where ndvi_data is NaN, otherwise 0
    cloudmask_data[np.isnan(ndvi_data)] = 1

    check_nan_cloudmask(ndvi_data, cloudmask_data)

    # Find indices where cloudmask == 0
    cloud_indices = np.argwhere(cloudmask_data == 0)
    # print(f"CLOUD INDICES ================== {cloud_indices}")
    n_total_clouds = len(cloud_indices)
    n_subset = int(cloud_percentage * n_total_clouds)

    # Randomly select 10% of the cloud indices to be set to 2
    if n_subset > 0:
        selected_indices = cloud_indices[np.random.choice(n_total_clouds, n_subset, replace=False)]
        for idx in selected_indices:
            cloudmask_data[tuple(idx)] = 2

    x, y, time = cloudmask_data.shape
    cloudmask_data = cloudmask_data.reshape(x * y, time)
    # print(cloudmask_data[0])

    return cloudmask_data







def process_ndvi(folder_path, output_path, output_cloudmask):
    # Dates with no data
    missing_dates = ['2023-01-05', '2023-02-06', '2023-02-14', '2023-03-26',
                    '2023-04-03', '2023-04-11', '2023-04-19', '2023-04-27']
    missing_dates = set(missing_dates)  # Convert to set for quick lookups

    # List of all available dates
    available_files = [f for f in os.listdir(folder_path) if f.endswith('.tif')]
    available_dates = sorted(
        datetime.strptime(f.split('_')[1].split('.')[0], "%Y-%m-%d") for f in available_files
    )

    # Check raster dimensions from a sample file
    with rasterio.open(os.path.join(folder_path, available_files[0])) as src:
        height, width = src.shape

    # Initialize list to hold time series data
    time_series = []

    # Generate the complete list of dates, assuming an 8-day interval
    start_date = available_dates[0]
    end_date = available_dates[-1]
    current_date = start_date

    while current_date <= end_date:
        date_str = current_date.strftime("%Y-%m-%d")
        file_path = os.path.join(folder_path, f"ndvi8days_{date_str}.tif")
        
        if date_str in missing_dates or not os.path.exists(file_path):
            # Append a null array for missing dates
            null_array = np.full((height, width), np.nan)

            # print(f"NULL ARRAY: {null_array.shape}")

            time_series.append(null_array)
        else:
            # Read and store data for available dates
            with rasterio.open(file_path) as src:
                time_series.append(src.read(1))  # Reads the first band

        # Increment by 8 days
        current_date += timedelta(days=8)

    # Stack along the time dimension, then transpose to (x, y, time)
    ndvi_data = np.stack(time_series, axis=0).transpose(1, 2, 0)

    cloudmask = process_cloudmask(ndvi_data)

    print(f"cloudmask : {cloudmask.shape}")
    # save cloudmask as .npy file
    np.save(output_cloudmask, cloudmask)
    print(f"cloudmask saved to {output_cloudmask}")

    # reshape ndvi and save  as .npy file
    x, y, time = ndvi_data.shape
    ndvi_data = ndvi_data.reshape(x * y, time)

    print(f"ndvi after reshape {ndvi_data.shape}")

    np.save(output_path, ndvi_data)
    
    print(f"Data saved to {output_path}")





def create_areamask(rvi_data, vh_data):
    """
    Create an area mask based on RVI and VH data. The mask is initialized to 1,
    and set to 0 for pixels where RVI or VH data is NaN.

    Parameters:
    rvi_data (numpy.ndarray): RVI data reshaped to (x * y, time).
    vh_data (numpy.ndarray): VH data reshaped to (x * y, time).

    Returns:
    numpy.ndarray: Area mask with shape (x * y,) where 0 indicates invalid pixels.
    """
    # Initialize area mask with 1s
    area_mask = np.ones(rvi_data.shape[0], dtype=np.int8)

    # Check for NaN in RVI or VH data and set corresponding area_mask to 0
    nan_rvi_indices = np.isnan(rvi_data)
    nan_vh_indices = np.isnan(vh_data)

    # print(f"============ {nan_rvi_indices}")

    # Update area_mask: set to 0 where RVI or VH data is NaN
    area_mask[np.any(nan_rvi_indices | nan_vh_indices, axis=1)] = 0

    count_valid_pixels = np.sum(area_mask == 1)

    print(f"Số phần tử bằng 1 trong areamask: {count_valid_pixels}")

    return area_mask





def process_sar(folder_path, output_rvi_path, output_vh_path, output_area_mask):
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

    print(f"rvi: {rvi_data.shape}")
    print(f"vh: {vh_data.shape}")

    # print(f"rvi after: {rvi_data}")

    x, y, time = rvi_data.shape
    rvi_data = rvi_data.reshape(x * y, time)
    vh_data = vh_data.reshape(x * y, time)

    # print(f"rvi after reshape: {rvi_data}")

    # Save each time series as .npy file
    np.save(output_rvi_path, rvi_data)
    np.save(output_vh_path, vh_data)

    print(f"RVI data saved to {output_rvi_path}")
    print(f"VH data saved to {output_vh_path}")

    areamask = create_areamask(rvi_data, vh_data)
    print(f"areamask: {areamask.shape}")
    
    np.save(output_area_mask, areamask)




dir = "/mnt/storage/huyekgis/brios/data2/"

ndvi_8days_folder = "ndvi_8days/"
output_ndvi_file = "numpy_data/ndvi.npy"
output_cloudmask_file = "numpy_data/cloudmask.npy"

rvi_8days_folder = "rvi_8days/"
output_rvi_file = "numpy_data/rvi.npy"

vh_8days_folder = "rvi_8days/"
output_vh_file = "numpy_data/vh.npy"

output_areamask_file = "numpy_data/areamask.npy"


process_ndvi(dir + ndvi_8days_folder, dir + output_ndvi_file, dir + output_cloudmask_file)
process_sar(dir + rvi_8days_folder, dir + output_rvi_file, dir + output_vh_file, dir + output_areamask_file)
