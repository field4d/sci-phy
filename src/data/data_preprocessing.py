import pandas as pd
import numpy as np
from scipy.signal import savgol_filter
from scipy.stats import linregress
import logging

# Set up global logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ----------------------------------------
# Data Preprocessing
# ----------------------------------------

def initial_data_preparation(file_path):
    """
    Initial preprocessing for the dataset by performing the following steps:
    1. Read data from a Parquet file.
    2. Rename 'index' column to 'timestamp'.
    3. Convert 'timestamp' column to datetime.
    4. Sort data by 'plant_ID' and 'timestamp'.
    5. Set 'timestamp' as the index.
    6. Create a 'unique_id' column by combining 'plant_ID', 'exp_ID', and 'gh_ID'.

    Args:
        file_path (str): Path to the Parquet file containing the data.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.

    Raises:
        ValueError: If required columns are missing.
        Exception: For any other issues during processing.
    """
    try:
        logger.info("Starting preprocessing for file: %s", file_path)
        data = pd.read_parquet(file_path)

        required_columns = {'index', 'plant_ID', 'exp_ID', 'gh_ID'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"Missing required columns: {required_columns - set(data.columns)}")

        logger.info("Renaming 'index' to 'timestamp'")
        data.rename(columns={"index": "timestamp"}, inplace=True)
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.sort_values(by=['plant_ID', 'timestamp'], inplace=True)
        data.set_index('timestamp', inplace=True)

        logger.info("Creating 'unique_id' by combining 'plant_ID', 'exp_ID', and 'gh_ID'")
        data['unique_id'] = data['plant_ID'].astype(str) + "_" + data['exp_ID'].astype(str) + "_" + data['gh_ID'].astype(str)

        unique_count = data['unique_id'].nunique()
        logger.info("Preprocessing completed. Number of unique IDs: %d", unique_count)

        return data
    except Exception as e:
        logger.error("Error during preprocessing: %s", str(e))
        raise

def clean_duplicated_timestamps(data: pd.DataFrame, unique_id_col: str = 'unique_id') -> pd.DataFrame:
    """
    Cleans a DataFrame by identifying and removing duplicated timestamps for each unique ID.

    Parameters:
    ------------
    data : pd.DataFrame
        The DataFrame containing plant data with timestamps as the index.
    unique_id_col : str, optional
        The name of the column containing unique plant identifiers. Default is 'unique_id'.

    Returns:
    --------
    pd.DataFrame
        A cleaned DataFrame with duplicated timestamps removed for each unique ID.

    Notes:
    ------
    - Keeps the first occurrence of duplicated timestamps.
    - Handles errors gracefully with logging.
    """
    try:
        # Validate input DataFrame
        if unique_id_col not in data.columns:
            raise ValueError(f"Column '{unique_id_col}' not found in the DataFrame.")

        for uid, plant_data in data.groupby(unique_id_col):
            try:
                if not pd.api.types.is_datetime64_any_dtype(plant_data.index):
                    logging.warning(f"UID {uid} has non-datetime timestamps.")
                elif plant_data.index.duplicated().any():
                    logging.info(f"UID {uid} has duplicated timestamps. Removing duplicates...")

                    # Drop duplicated timestamps, keeping the first occurrence
                    cleaned_plant_data = plant_data[~plant_data.index.duplicated(keep='first')]

                    # Remove the original data for the current UID
                    data = data[data[unique_id_col] != uid]
                    # Append the cleaned data back
                    data = pd.concat([data, cleaned_plant_data])
            
            except Exception as e:
                logging.error(f"Error processing UID {uid}: {e}")

        logging.info("Duplicate cleaning process completed successfully.")
        return data

    except Exception as e:
        logging.critical(f"Critical error during duplicate cleaning: {e}")
        return data  # Return the original data if an error occurs
    

# ----------------------------------------
# Outlier Handling
# ----------------------------------------

def outliers_by_thresholds(data, column, thresholds):
    """
    Replace outliers in a column with NaN based on thresholds.

    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): Column to check for outliers.
        thresholds (list): [lower, upper] thresholds or [upper] for one-sided.

    Returns:
        pd.DataFrame: Updated DataFrame with outliers replaced by NaN.
    """
    try:
        logger.info(f"Processing outliers for column '{column}' with thresholds: {thresholds}")
        if len(thresholds) == 2:
            lower, upper = thresholds
            condition = (data[column] < lower) | (data[column] > upper)
        elif len(thresholds) == 1:
            condition = data[column] > thresholds[0]
        else:
            logger.warning("No thresholds provided. Skipping column: %s", column)
            return data

        data.loc[condition, column] = np.nan
        logger.info("Outliers set to NaN in column: %s", column)
        return data
    except Exception as e:
        logger.error(f"Error in set_outliers_to_nan for column '{column}': {e}")
        raise

def outliers_with_moving_avg(df, value_column, window_size=30, threshold=2.5):
    """
    Finds outliers in a DataFrame based on a moving average and replaces them with NaN,
    except for points between 20:00 and 4:00.

    Args:
        df (pd.DataFrame): DataFrame containing the data with a datetime index.
        value_column (str): Name of the column containing values to process.
        window_size (int): Size of the moving average window.
        threshold (float): Number of standard deviations to flag outliers.

    Returns:
        pd.DataFrame: Updated DataFrame with outliers replaced by NaN
    """
    try:
        logger.info(f"Finding outliers for column: {value_column} using moving average with window size {window_size} and threshold {threshold}")
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DatetimeIndex.")

        df['moving_avg'] = df[value_column].rolling(window=window_size, center=True, min_periods=10).mean()
        df['moving_std'] = df[value_column].rolling(window=window_size, center=True, min_periods=10).std()

        df['is_outlier'] = (np.abs(df[value_column] - df['moving_avg']) > threshold * df['moving_std'])

        df['hour'] = df.index.hour
        df['exclude'] = ((df['hour'] >= 20) | (df['hour'] < 4))
        df[f'{value_column}_outlier'] = np.where(df['is_outlier'] & ~df['exclude'], True, False)

        df[value_column] = np.where(df[f'{value_column}_outlier'], np.nan, df[value_column])

        if 'unique_id' in df.columns:
            outlier_plants = df.loc[df[f'{value_column}_outlier'], 'unique_id'].unique()
            if len(outlier_plants) > 0:
                logger.info(f"Outliers found in the following plant IDs: {outlier_plants}")

        df.drop(['moving_avg', 'moving_std', 'hour','is_outlier','exclude'], axis=1, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error in find_outliers_with_moving_avg: {str(e)}")
        raise

# ----------------------------------------
# Missing Value Interpolation
# ----------------------------------------

def interpolate_missing_values(data, columns, plant_id_col='unique_id',timestamp_column='timestamp', method='linear', threshold=40):
    """
    Interpolates missing values within each unique plant's data.

    Args:
        data (pd.DataFrame): Input DataFrame with missing values.
        columns (list): List of columns to interpolate.
        plant_id_col (str): Name of the column identifying each unique plant (default is 'unique_id').
        timestamp_column (str): Name of the column containing timestamp data (default is 'timestamp')
        method (str): Interpolation method (default: 'linear'). Other options include 'cubic'.
        threshold (int): Max consecutive NaNs allowed for interpolation.

    Returns:
        pd.DataFrame: DataFrame with interpolated values for each plant and column,
        with gaps larger than the threshold left as NaN.
    """
    try:
        logger.info(f"Interpolating missing values for columns: {columns} using method: {method}")
        
        # Ensure timestamp column is the index
        if timestamp_column not in data.index.names:
            data = data.set_index(timestamp_column)
        
        # Resulting DataFrame with interpolated values
        interpolated_data = []

        # Loop through each unique plant ID and interpolate its data
        for uid in data[plant_id_col].unique():
            plant_data = data.loc[data[plant_id_col] == uid].copy()
            for col in columns:
                if plant_data[col].isna().sum() > 0 and not plant_data[col].isna().all():  # Only interpolate if there are NaNs but not if all values are NaNs
                    # Calculate gaps and their indices in plant_data
                    gap_sizes = plant_data[col].isnull().astype(int).groupby(
                        (plant_data[col].notnull().astype(int).cumsum())
                    ).transform('size') * plant_data[col].isnull().astype(int)

                    # Mask for indices in large gaps
                    large_gap_mask = gap_sizes > threshold
                    # Interpolate only for indices not in large gaps
                    plant_data.loc[~large_gap_mask, col] = plant_data.loc[
                        ~large_gap_mask, col
                    ].interpolate(method=method)

                    # Check if any NaN values remain in the new column
                    if plant_data[col].isna().any():
                        logger.info(f"Unique ID '{uid}' has {plant_data[col].isna().sum()} NaN values remaining in '{col}' after interpolation.")

            interpolated_data.append(plant_data)

        interpolated_data = pd.concat(interpolated_data)

        logger.info("Interpolation completed successfully.")
        return interpolated_data

    except Exception as e:
        logger.error(f"Error occurred during interpolation: {e}")
        raise

def drop_days_with_missing_s4(data, unique_id_col='unique_id', target_col='s4'):
    """
    Drops the first or last day for each unique ID if there is missing data in the 's4' column
    after 4:00 AM on the first day or before 8:00 PM on the last day.

    Args:
        data (pd.DataFrame): DataFrame with time-indexed data.
        unique_id_col (str): Column representing unique identifiers.
        target_col (str): The column to check for missing data.

    Returns:
        pd.DataFrame: Cleaned DataFrame with days removed if they contain missing 's4' data.
    """
    try:
        logger.info("Starting to drop days with missing 's4' values for each unique ID")
        
        # Ensure the DataFrame has a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DatetimeIndex.")
        
        cleaned_data = []

        for uid, group in data.groupby(unique_id_col):
            logger.info(f"Processing unique ID: {uid}")
            
            # Sort the group by timestamp
            group = group.sort_index()

            # Identify the first and last dates
            first_day = group.index.date[0]
            last_day = group.index.date[-1]

            # Check for missing data after 4 AM on the first day
            first_day_data = group[(group.index.date == first_day) & (group.index.time > pd.to_datetime('04:00:00').time())]
            if first_day_data[target_col].isnull().any():
                logger.info(f"Dropping first day for unique ID {uid} due to missing 's4' values after 4 AM")
                group = group[group.index.date != first_day]

            # Check for missing data before 8 PM on the last day
            last_day_data = group[(group.index.date == last_day) & (group.index.time < pd.to_datetime('20:00:00').time())]
            if last_day_data[target_col].isnull().any():
                logger.info(f"Dropping last day for unique ID {uid} due to missing 's4' values before 8 PM")
                group = group[group.index.date != last_day]

            cleaned_data.append(group)

        result = pd.concat(cleaned_data)
        logger.info("Dropping of days with missing 's4' values completed successfully.")
        return result

    except Exception as e:
        logger.error(f"Error in drop_days_with_missing_s4: {e}")
        raise

def moving_average_with_kernel_pandas(data, column, window_size, kernel_type='gaussian', std_dev=1):
    """
    Applies a moving average with a specified kernel to a column in a DataFrame using pandas' rolling functionality.

    Args:
        data (pd.DataFrame): Input DataFrame.
        column (str): Column to smooth.
        window_size (int): Size of the moving window.
        kernel_type (str): Kernel type ('gaussian', 'triangular', etc.).
        std_dev (float): Standard deviation for Gaussian kernel.

    Returns:
        pd.DataFrame: Updated DataFrame with smoothed values.
    """
    try:
        logger.info(f"Applying {kernel_type} smoothing on column: {column} with window size {window_size}")
        if kernel_type == 'gaussian':
            data[column] = data[column].rolling(window=window_size, win_type='gaussian', center=True, min_periods=1).mean(std=std_dev)
        else:
            data[column] = data[column].rolling(window=window_size, win_type=kernel_type, center=True, min_periods=1).mean()
        return data
    except Exception as e:
        logger.error(f"Error in moving_average_with_kernel_pandas: {str(e)}")
        raise
# ----------------------------------------
# Plant weight column preprocessing
# ----------------------------------------

def smooth_data(data, window_length=9, polyorder=2):
    """
    Apply Savitzky-Golay filter to smooth the data.

    Parameters:
    data (pd.Series): The data to be smoothed.
    window_length (int): The length of the filter window (must be odd and greater than polyorder).
    polyorder (int): The order of the polynomial used to fit the samples.

    Returns:
    pd.Series: Smoothed data.
    """
    try:
        if len(data) < window_length:
            window_length = len(data) - (len(data) % 2) - 1
        if window_length > 2:
            return savgol_filter(data, window_length, polyorder, mode='nearest')
        else:
            return data
    except Exception as e:
        logging.error(f"Error in smooth_data: {e}")
        return data

def process_group_weight(group):
    """
    Process a group of plant data to clean, shift, interpolate, and smooth the weight values.

    Parameters:
    group (pd.DataFrame): A group of plant data.

    Returns:
    pd.DataFrame: Processed plant data.
    """
    try:
        group = group.copy()
        group['pnw'] = group['pnw'].apply(lambda x: x if x >= 3 else np.nan)

        if group['pnw'].iloc[:4].isnull().all() or group['pnw'].max() > 1500:
            first_valid_index = group['pnw'].first_valid_index()
            if first_valid_index is not None:
                group['pnw'] = group['pnw'] - group.loc[first_valid_index, 'pnw'] + 10
            else:
                group['pnw'] = np.nan
                logger.info(f"No plant weight data for unique_id: {group['unique_id'].iloc[0]}")

        group['pnw'] = group['pnw'].cummax()
        group['pnw'] = group['pnw'].interpolate(method='linear', limit_direction='both')
        group['pnw'] = smooth_data(group['pnw'])

        return group
    except Exception as e:
        logging.error(f"Error in process_group: {e}")
        return group

def process_weight_data(data):
    """
    Main function to process plant data by grouping, cleaning, interpolating, and smoothing.

    Parameters:
    data (pd.DataFrame): The input data containing plant information.

    Returns:
    pd.DataFrame: The processed data with smoothed plant weights.
    """
    try:
        data = data.copy()
        pnw_df = data[['dt', 'pnw', 'unique_id']]
        pnw_df = pnw_df[pnw_df.index.time == pd.to_datetime('00:00:00').time()]

        pnw_df.sort_values(by=['unique_id', data.index.name], inplace=True)
        grouped = pnw_df.groupby('unique_id', group_keys=False)

        pnw_df_processed = grouped.apply(process_group_weight)

        df_reset = data.reset_index()
        df_reset = df_reset.drop(columns=['pnw'], errors='ignore') #drop the original pnw
        pnw_df_processed = pnw_df_processed.reset_index()
        df_merged = df_reset.merge(pnw_df_processed[['timestamp', 'unique_id', 'pnw']], on=['timestamp', 'unique_id'], how='left')

        df_merged.set_index('timestamp', inplace=True)
        logging.info("Weight data processing completed successfully.")
        return df_merged
    
    except Exception as e:
        logging.error(f"Error in process_weight_data: {e}")
        return data



# ----------------------------------------
# Growth Calculation
# ----------------------------------------

def calculate_growth(data, condition_col='condition', target_col='s4'):
    """
    Calculate the growth (slope) for each unique_id based on days where the condition is 'W'.
    If a plant has only one condition (e.g., all 'B' or all 'C'), calculate the growth for those days.

    Args:
        data (pd.DataFrame): Input DataFrame.
        condition_col (str): The column representing the condition (e.g., 'W', 'B', 'C').
        target_col (str): The column to calculate growth on (e.g., 's4').

    Returns:
        pd.DataFrame: DataFrame with an added 'growth' column.
    """
    try:
        logger.info("Calculating growth for each unique_id based on conditions")
        growth_values = []

        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("The DataFrame index must be a DatetimeIndex.")

        for uid, group in data.groupby('unique_id'):
            group = group.sort_index()
            
            # Determine conditions present for the unique_id
            conditions = group[condition_col].unique()

            if 'W' in conditions:
                logger.info(f"Calculating growth for unique_id {uid} on 'W' condition days")
                filtered_group = group[group[condition_col] == 'W']
            elif len(conditions) == 1:
                logger.info(f"Calculating growth for unique_id {uid} with constant condition: {conditions[0]}")
                filtered_group = group
            else:
                logger.warning(f"Skipping unique_id {uid} due to mixed conditions without 'W'")
                growth_values.append((uid, np.nan))
                continue

            # Filter between 4:00 and 5:00 AM
            filtered_group = filtered_group.between_time("04:00", "05:00")

            if filtered_group.empty:
                logger.warning(f"No data for unique_id {uid} after filtering. Skipping.")
                slope = np.nan
            elif len(filtered_group) > 1:
                slope, _, _, _, _ = linregress(
                    filtered_group.index.map(pd.Timestamp.timestamp),
                    filtered_group[target_col]
                )
                logger.info(f"Calculated slope for unique_id {uid}: {slope}")
            else:
                logger.warning(f"Not enough points for slope calculation for unique_id {uid}. Setting NaN.")
                slope = np.nan

            growth_values.append((uid, slope))

        growth_df = pd.DataFrame(growth_values, columns=['unique_id', 'growth'])

        data = data.reset_index()
        data = data.merge(growth_df, on='unique_id', how='left')
        data.set_index('timestamp', inplace=True)

        return data
    except Exception as e:
        logger.error(f"Error in calculate_growth: {str(e)}")
        raise

