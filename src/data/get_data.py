import os
import pandas as pd
import requests
import logging
import json
import numpy as np
from datetime import timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Securely load API credentials from environment variables
AUTHORIZATION = os.getenv('AUTHORIZATION')

# 1. Load Metadata
def load_metadata(file_path):
    """
    Load metadata from an Excel file.

    Parameters:
        file_path (str): Path to the Excel file.

    Returns:
        pd.DataFrame: Loaded metadata as a DataFrame.
    """
    try:
        logging.info(f"Loading metadata from {file_path}")
        meta_data = pd.read_csv(file_path)
        logging.info(f"Metadata loaded successfully. Here are the first few rows:\n{meta_data.head()}")
        return meta_data
    
    except Exception as e:
        logging.error(f"Error loading metadata: {e}")
        raise

# 2. Clean Metadata
def clean_metadata(df, columns_to_drop=None, rename_dict=None):
    """
    Clean and preprocess metadata by:
    1. Filtering rows where 'Keep' == 'Yes'
    2. Dropping unnecessary columns
    3. Renaming columns for consistency

    Parameters:
        df (pd.DataFrame): Raw metadata.
        columns_to_drop (list, optional): Columns to be dropped. Defaults to common metadata columns.
        rename_dict (dict, optional): Mapping for renaming columns. Defaults to standard naming.

    Returns:
        pd.DataFrame: Cleaned metadata.
    """
    # Default configurations
    if columns_to_drop is None:
        columns_to_drop = ['Info', 'Unseen by User?', 'Daily Check']
    if rename_dict is None:
        rename_dict = {
            'Greenhouse': 'controlId',
            'Experiment ID': 'expId',
            'Plant ID(s)': 'plants',
            'Keep': 'keep',
            'Treatment': 'condition',
            'Start Date': 'fromDate',
            'Treatment Start Date': 'TreatStart',
            'End Date': 'toDate',
            'Remove Dates': 'ToRemove',
            'Remove Date 1': 'ToRemove1',
            'Remove Date 2': 'ToRemove2',
            'Crop Type': 'plant_type',
            'Crop Name': 'plant_name'
        }

    try:
        logging.info("Starting metadata cleaning process")

        # Filter rows where 'Keep' == 'Yes'
        if 'Keep' in df.columns:
            df = df[df['Keep'].str.contains("Yes", na=False)]
            logging.info(f"Filtered metadata to {len(df)} rows with 'Keep' = 'Yes'")
        else:
            logging.warning("'Keep' column not found in DataFrame")

        # Drop unnecessary columns
        columns_to_drop_actual = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=columns_to_drop_actual)
        logging.info(f"Dropped columns: {columns_to_drop_actual}")

        # Rename columns
        rename_dict_actual = {k: v for k, v in rename_dict.items() if k in df.columns}
        df = df.rename(columns=rename_dict_actual)
        logging.info(f"Renamed columns: {rename_dict_actual}")

        # Reset index
        df = df.reset_index(drop=True)
        logging.info("Metadata cleaning process completed successfully")

        return df

    except Exception as e:
        logging.error(f"Error during metadata cleaning: {e}")
        raise


def standardize_metadata(df):
    """
    Standardize metadata including date formats, ID mappings, conditions, and data types.

    Parameters:
        df (pd.DataFrame): The metadata DataFrame.

    Returns:
        pd.DataFrame: Standardized DataFrame.
    """
    logging.info("Starting metadata standardization.")

    try:
        # 1 Control ID Mapping
        control_id_dict = {"Icore": 3, "Lizzy": 33, "Room101": 23, "ISURF": 60, "Moris": 42}
        if 'controlId' in df.columns:
            df['controlId'] = df['controlId'].replace(control_id_dict)
            logging.info("Mapped control IDs.")
            logging.info(f"Unique control IDs after mapping: {df['controlId'].unique()}")

        # 2 Ensure expId is Integer
        if 'expId' in df.columns:
            df['expId'] = pd.to_numeric(df['expId'], errors='coerce').fillna(0).astype(int)
            logging.info("Converted 'expId' to integer.")

        # 3 Condition Mapping
        condition_dict = {
            "Control": "W",
            "Cutting": "C",
            "Drought - Terminal": "DT",
            "Drought - With some irrigation (less than 100 gr)": "DP",
            "Salt": "S",
            "Biotic stress": "B",
            "Not relevant": "NR"
        }
        if 'condition' in df.columns:
            df['condition'] = df['condition'].replace(condition_dict)
            logging.info("Standardized 'condition' values.")
            logging.info(f"Unique conditions after mapping: {df['condition'].unique()}")

        # 4 Date-Time Standardization
        End_of_day = pd.Timedelta(hours=23, minutes=59, seconds=59, milliseconds=999)

        date_columns = ['fromDate', 'toDate', 'TreatStart', 'ToRemove1', 'ToRemove2']
        for col in date_columns:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)  # Ensure datetime format
                if col == 'toDate':
                    df[col] = df[col] + End_of_day  # Add EOD time for toDate
        logging.info(f"Standardized date formats for columns: {date_columns}")

        logging.info("Metadata standardization completed successfully.")
        return df

    except Exception as e:
        logging.error(f"Error during metadata standardization: {e}")
        raise

# 4. Split Plant Numbers
def split_plant_numbers(df, column_name='plants'):
    """
    Split multiple plant numbers in a single cell into separate rows.

    Parameters:
        df (pd.DataFrame): DataFrame containing plant data.
        column_name (str): Name of the column to split.

    Returns:
        pd.DataFrame: DataFrame with individual plant numbers in separate rows.
    """
    logging.info("Starting to split plant numbers.")

    try:
        if column_name not in df.columns:
            logging.warning(f"Column '{column_name}' not found in the DataFrame.")
            return df  # Return the original DataFrame if the column doesn't exist

        new_df = pd.DataFrame(columns=df.columns)
        total_new_rows = 0

        for index, row in df.iterrows():
            # Split the plant numbers (assumed to be space-separated)
            plant_numbers = str(row[column_name]).split()

            logging.debug(f"Row {index}: Found plant numbers {plant_numbers}")

            for number in plant_numbers:
                new_row = row.copy()  # Maintain original data
                new_row[column_name] = number
                new_df = pd.concat([new_df, pd.DataFrame([new_row])], ignore_index=True)
                total_new_rows += 1

        logging.info(f"Splitting complete. Original rows: {len(df)}, New rows: {total_new_rows}")
        return new_df

    except Exception as e:
        logging.error(f"Error while splitting plant numbers: {e}")
        raise

#5. Adjust dates
def adjust_dates(df, column_name = 'ToRemove'):
    """
    Adjusts the 'fromDate' and 'toDate' based on the 'ToRemove', 'ToRemove1', and 'ToRemove2' columns.
    Duplicates rows where necessary and modifies the 'fromDate' and 'toDate' accordingly.
    Also includes rows where 'ToRemove' is 'No' without any changes.

    Args:
    df (DataFrame): The DataFrame to process.
    column_name (str): The name of the column that contains the 'ToRemove' criteria.

    Returns:
    DataFrame: A new DataFrame with adjusted dates and duplicated rows where necessary, 
               and rows with 'ToRemove' as 'No' left unchanged.
    """
    
    logging.info("Starting date adjustment process.")
    
    try:
        # Validate required columns
        required_columns = ['ToRemove', 'ToRemove1', 'toDate']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logging.warning(f"Missing columns in DataFrame: {missing_columns}")
            return df  # Return original DataFrame if essential columns are missing
        
        # Create an empty DataFrame to store the processed rows
        new_df = pd.DataFrame(columns=df.columns)
        # Iterate over each row in the DataFrame
        for index, row in df.iterrows():
            # Handle the case where 'ToRemove' contains 'Specific' and handle accordingly
            if 'Specific' in row[column_name] and pd.notnull(row['ToRemove1']):
                # Case 1: Adjust 'toDate' to the day before 'ToRemove1'
                new_row_1 = row.copy()
                new_row_1['toDate'] = row['ToRemove1'] - timedelta(milliseconds=1)
                new_df = pd.concat([new_df, pd.DataFrame([new_row_1])], ignore_index=True)
                
                # Case 2: Adjust 'fromDate' after 'ToRemove1' and handle 'ToRemove2'
                if pd.notnull(row['ToRemove2']):
                    new_row_2 = row.copy()
                    new_row_2['fromDate'] = row['ToRemove1'] + timedelta(days=1)
                    new_row_2['toDate'] = row['ToRemove2'] - timedelta(milliseconds=1)
                    new_df = pd.concat([new_df, pd.DataFrame([new_row_2])], ignore_index=True)
                    
                    new_row_3 = row.copy()
                    new_row_3['fromDate'] = row['ToRemove2'] + timedelta(days=1)
                    new_df = pd.concat([new_df, pd.DataFrame([new_row_3])], ignore_index=True)
                else:
                    new_row_2 = row.copy()
                    new_row_2['fromDate'] = row['ToRemove1'] + timedelta(days=1)
                    new_df = pd.concat([new_df, pd.DataFrame([new_row_2])], ignore_index=True)
            
            # Handle the case where 'ToRemove' contains 'Span'
            elif 'Span' in row[column_name] and pd.notnull(row['ToRemove1']):
                # Case 1: Adjust 'toDate' to the day before 'ToRemove1'
                new_row_1 = row.copy()
                new_row_1['toDate'] = row['ToRemove1'] - timedelta(milliseconds=1)
                new_df = pd.concat([new_df, pd.DataFrame([new_row_1])], ignore_index=True)
                
                # Case 2: Adjust 'fromDate' after 'ToRemove2', if it exists
                if pd.notnull(row['ToRemove2']):
                    new_row_2 = row.copy()
                    new_row_2['fromDate'] = row['ToRemove2'] + timedelta(days=1)
                    new_df = pd.concat([new_df, pd.DataFrame([new_row_2])], ignore_index=True)
            
            # If 'ToRemove' is 'No', simply append the original row without any changes
            else:
                new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)

        for col in ['fromDate', 'toDate']:
            df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)  # Ensure datetime format

        logging.info(f"Date adjustment process completed successfully. Total adjusted rows: {len(new_df) - len(df)}")
        return new_df

    except Exception as e:
        logging.error(f"Unexpected error during date adjustment: {e}")
        raise

# 6. Test metadata
def check_duration_anomalies(df):
    """
    Identify plants with unexpected duration (<0 or >70 days).
    """
    df['duration_days'] = (df['toDate'] - df['fromDate']).dt.days
    filtered_plants = df[(df['duration_days'] < 0) | (df['duration_days'] > 70)]
    
    if not filtered_plants.empty:
        logging.warning(f"Plants with unexpected duration: {filtered_plants['plants'].unique()}")
    else:
        logging.info("No duration anomalies detected.")
    
    return filtered_plants


def check_missing_dates(df):
    """
    Identify plants with missing 'fromDate' or 'toDate'.
    """
    missing_dates = df[df['fromDate'].isna() | df['toDate'].isna()]
    
    if not missing_dates.empty:
        logging.warning(f"Plants with missing dates: {missing_dates['plants'].tolist()}")
    else:
        logging.info("No missing 'fromDate' or 'toDate' detected.")
    
    return missing_dates


def check_overlapping_dates(df):
    """
    Identify overlapping date ranges for each unique plant.
    """
    df_copy = df.copy()
    df_copy['fromDate'] = pd.to_datetime(df_copy['fromDate'], errors='coerce')
    df_copy['toDate'] = pd.to_datetime(df_copy['toDate'], errors='coerce')
    
    df_copy = df_copy.sort_values(by=['plants', 'fromDate']).reset_index(drop=True)
    overlaps = []

    for plant_id in df_copy['plants'].unique():
        plant_data = df_copy[df_copy['plants'] == plant_id]
        
        for i in range(len(plant_data) - 1):
            current_to = plant_data.iloc[i]['toDate']
            next_from = plant_data.iloc[i + 1]['fromDate']
            
            if pd.notnull(current_to) and pd.notnull(next_from) and next_from <= current_to:
                overlaps.append(plant_data.iloc[i])
                overlaps.append(plant_data.iloc[i + 1])

    if overlaps:
        logging.warning(f"Overlapping dates found for plants: {pd.DataFrame(overlaps)['plants'].unique()}")
        return pd.DataFrame(overlaps).drop_duplicates()
    else:
        logging.info("No overlapping dates detected.")
        return pd.DataFrame(columns=df.columns)


def test_metadata(df):
    """
    Run all metadata tests:
    1. Duration anomalies
    2. Missing dates
    3. Overlapping date ranges
    """
    logging.info("Starting metadata validation tests.")
    
    duration_anomalies = check_duration_anomalies(df)
    missing_dates = check_missing_dates(df)
    overlapping_dates = check_overlapping_dates(df)

    issues_detected = {
        'duration_anomalies': duration_anomalies,
        'missing_dates': missing_dates,
        'overlapping_dates': overlapping_dates
    }
    
    logging.info("Metadata validation completed.")
    logging.info(f"Final Metadata first few rows:\n{df.head()}")
    return issues_detected


# 7. Fetch Data from API
# Check Soil or Sand
def check_soil_sand(df, threshold=4700):
    """
    Determines if the plant is grown in soil or sand based on the initial weight ('s4' column).
    
    Args:
        df (DataFrame): Data containing the 's4' column (weight measurements).
        threshold (int, optional): Threshold to classify between soil and sand. Default is 4700.
    
    Returns:
        str: 'sand' if median weight exceeds threshold, 'soil' if below threshold,
             or 'unknown' if there is insufficient data.
    """
    # Check if 's4' exists and has non-NaN data
    if 's4' not in df.columns or df['s4'].iloc[:1440].dropna().empty:
        logging.warning("Insufficient data in the first 3 days of 's4' to determine soil/sand.")
        return 'unknown'
    
    if 'controlId' == 42 or df['s4'].iloc[:1440].median() >= 10000:
        logging.warning("Large pots, can't infer soil type by pot weight")
        return 'unknown'
    
    # Calculate median and classify
    median_weight = df['s4'].iloc[:1440].median()
    is_sand = median_weight > threshold

    logging.info(f"Median weight: {median_weight}, Classified as: {'sand' if is_sand else 'soil'}")
    
    return 'sand' if is_sand else 'soil'


# Pull Data from SPAC API
def fetch_data_from_SPAC(start_date, end_date, authorization,
                        plants_id, exp_id, control_id):
    """
    Fetches raw data from the SPAC API.
    
    Args:
        start_date (datetime): Start date.
        end_date (datetime): End date.
        authorization (str): API token.
        plants_id (str): Plant ID.
        exp_id (str): Experiment ID.
        control_id (str): Control system ID.

    Returns:
        dict: Dictionary containing raw API data or None if request fails.
    """
    logging.info(f"Fetching data for exp: {exp_id}, plant: {plants_id}, control: {control_id}")
    payload={}
    headers = {'Authorization': f'{authorization}'}
    parameters = "s4,wthrsrh,wthrstemp,wthrspar,wthrsvpd"
    daily_parameters = "dt,pnw"

    start_date_f = start_date.strftime('%Y-%m-%dT%H:%M:%S')
    end_date_f = (end_date + timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%S')

    # URLs
    daily_url = f"https://spac.plant-ditech.com/api/data/getData?experimentId={exp_id}&controlSystemId={control_id}&fromDate={start_date_f}&toDate={end_date_f}&plants={plants_id}&params={daily_parameters}"
    continuous_url = f"https://spac.plant-ditech.com/api/data/getData?experimentId={exp_id}&controlSystemId={control_id}&fromDate={start_date_f}&toDate={end_date_f}&plants={plants_id}&params={parameters}"

    try:
        # Fetch Daily Data
        logging.info(f"Fetching daily data from: {daily_url}")
        daily_response = requests.get(daily_url, headers=headers, data=payload)

        if daily_response.status_code == 500:
            logging.critical("One or more of your inputs are incorrect, or the server is down.")
            return None
        if daily_response.status_code != 200:
            logging.error(f"Daily request failed: {daily_response.status_code}")
            return None
        daily_data = daily_response.json()

        # Fetch Continuous Data
        logging.info(f"Fetching continuous data from: {continuous_url}")
        continuous_response = requests.get(continuous_url, headers=headers, data=payload)
        if continuous_response.status_code != 200:
            logging.error(f"Continuous request failed: {continuous_response.status_code}")
            return None
        continuous_data = continuous_response.json()

        return {"daily": daily_data, "continuous": continuous_data}

    except requests.exceptions.RequestException as req_err:
        logging.error(f"Request error: {req_err}")
        return None
    
def process_SPAC_data(raw_data, start_date, end_date, TreatStart, plants_id, exp_id, control_id, plant_type=None, condition=None, soil_type=None):
    """
    Processes raw SPAC API data and converts it into structured DataFrames.
    
    Args:
        raw_data (dict): Raw API response data.
        start_date (datetime): Start date.
        end_date (datetime): End date.
        exp_id (str): Experiment ID.
        plant_id (str): Plant ID.
        plant_type (str, optional): Plant type.

    Returns:
        tuple: (processed continuous data, processed daily data)
    """
    if not raw_data:
        logging.error("No raw data provided for processing.")
        return None, None
    
    # Daily Data Processing
    daily_group_data = raw_data["daily"]['group1']
    # Create daily values DataFrame
    daily_tr = pd.DataFrame(daily_group_data['data']['dt'], columns=['timestamp', 'dt'])
    plant_weight = pd.DataFrame(daily_group_data['data']['pnw'], columns=['timestamp', 'pnw'])
    merged_daily_data = pd.merge(daily_tr, plant_weight, on='timestamp', how='outer')
    
    if daily_tr.empty or plant_weight.empty:
        logging.warning("Daily data is empty.")
        return None, None
    
    group_data = raw_data["continuous"]['group1']
    # Prepare data for 's4' and 'wsrh'
    s4_data = pd.DataFrame(group_data['data']['s4'], columns=['timestamp', 's4'])
    wsrh_data = pd.DataFrame(group_data['data']['wthrsrh'], columns=['timestamp', 'wsrh'])
    wstemp_data = pd.DataFrame(group_data['data']['wthrstemp'], columns=['timestamp', 'wstemp'])
    wspar_data = pd.DataFrame(group_data['data']['wthrspar'], columns=['timestamp', 'wspar'])
    wsvpd_data = pd.DataFrame(group_data['data']['wthrsvpd'], columns=['timestamp', 'vpd'])

    # Merging the four datasets with an outer merge
    merged_data = pd.merge(s4_data, wsrh_data, on='timestamp', how='outer')
    merged_data = pd.merge(merged_data, wstemp_data, on='timestamp', how='outer')
    merged_data = pd.merge(merged_data, wspar_data, on='timestamp', how='outer')
    merged_data = pd.merge(merged_data, wsvpd_data, on='timestamp', how='outer')
    
    # Timestamp organizing for further data adding
    merged_data['timestamp'] = pd.to_datetime(merged_data['timestamp'], format='%Y-%m-%dT%H:%M:%S')
    merged_data.set_index('timestamp', inplace=True)
    merged_data.sort_index(inplace=True) 
    
    # Add the missing timestamp check and insertion after sorting the index
    expected_interval = timedelta(minutes=3)
    all_timestamps = pd.date_range(start=start_date, end=end_date, freq=expected_interval)
    merged_data = merged_data.reindex(all_timestamps, fill_value=np.NaN)
    
    # add constent info
    merged_data['gh_ID'] = control_id
    merged_data['exp_ID'] = exp_id
    merged_data['plant_ID'] = group_data['plants'][0]
    merged_data['plant_type'] = plant_type
    if soil_type:
        merged_data['soil_sand'] = soil_type
    else:
        merged_data['soil_sand'] = check_soil_sand(merged_data, threshold=4700)
    
    # calculate weight change
    merged_data['Weight_change'] = merged_data['s4'].diff()
        
    # Update condition column befor treatment and in the treatment
    if not pd.isna(TreatStart): # If treatment start date is provided
        merged_data.loc[(merged_data.index >= start_date) &
                        (merged_data.index < TreatStart), 'condition'] = 'W'
        merged_data.loc[(merged_data.index >= TreatStart) &
                        (merged_data.index <= end_date), 'condition'] = condition 
    else:
        merged_data['condition'] = condition
        
    # Set time to 00:00:00 for daily data
    merged_daily_data['timestamp'] = pd.to_datetime(merged_daily_data['timestamp'])
    merged_daily_data['timestamp'] = merged_daily_data['timestamp'].dt.normalize()
    logging.debug(f"{merged_daily_data.head()}")

    merged_data = merged_data.reset_index() # Reset index so timestamp is in the table
    logging.debug(f"{merged_data.head()}")

    # Merge on the datetime (considering date with 00:00:00 time)
    final_merged_data = pd.merge(merged_data, merged_daily_data, left_on='index', right_on='timestamp', how='left')

    # Drop redundant columns if necessary
    final_merged_data.drop(columns=['timestamp'], inplace=True)
    logging.debug(f"{final_merged_data.head()}")
    
    logging.info(f"got the data of plant {plants_id}, exp {exp_id} from Date:{start_date} to {end_date}")
    return final_merged_data



# Append Data to Parquet
def append_to_parquet(plant_df, parquet_file, plants_id, exp_id):
    """
    Appends data to a Parquet file if it exists, or creates a new file if it doesn't.

    Args:
    plant_df (DataFrame): Data to be appended.
    parquet_file (str): Path to the Parquet file.
    plants_id (str or int): ID of the plant.
    exp_id (str or int): Experiment ID.
    remove_duplicates (bool): Optionally remove duplicates after appending (default is False).
    
    Returns:
    None
    """
    if os.path.exists(parquet_file): # Check if the Parquet file exists
        # Read the existing Parquet file
        existing_df = pd.read_parquet(parquet_file)
        combined_df = pd.concat([existing_df, plant_df], ignore_index=True)
        combined_df.to_parquet(parquet_file, index=False)
        logging.info(f"Finished appending plant {plants_id} from expId {exp_id}")    
    else:
        # If the file does not exist, create a new Parquet file with the DataFrame
        plant_df.to_parquet(parquet_file, index=False)
        logging.info(f"New Parquet file created for plant {plants_id} from expId {exp_id}")


# Main Workflow
def main(metadata_df, authorization, output_parquet):
    """
    Main data processing pipeline:
    - Iterates through metadata.
    - Pulls data from SPAC API.
    - Appends data to Parquet files.

    Args:
        metadata_df (DataFrame): Metadata with plant and experiment info.
        authorization (str): API authorization token.
        output_parquet (str): Path to the output Parquet file.

    Returns:
        None
    """
    for _, row in metadata_df.iterrows():
        try:
            raw_data = fetch_data_from_SPAC(row['fromDate'], row['toDate'], authorization, row['plants'], row['expId'], row['controlId'])

            if raw_data is None:
                logging.error("Failed to retrieve raw data.")
                return None

            plant_df = process_SPAC_data(raw_data,
                start_date=row['fromDate'],
                end_date=row['toDate'],
                TreatStart=row['TreatStart'],
                plants_id=row['plants'],
                exp_id=row['expId'],
                control_id=row['controlId'],
                plant_type=row['plant_type'],
                condition=row['condition']
            )

            if plant_df is None:
                logging.error("Processing step failed.")
                return None

            if not plant_df.empty:
                append_to_parquet(plant_df, output_parquet, row['plants'], row['expId'])

        except Exception as e:
            logging.error(f"Error processing plant {row['plants']} (exp {row['expId']}): {e}")

    logging.info("All data processing complete!")

if __name__ == "__main__":
    try:
        # Make sure your data was collected via the 'get_data_form.py'
        data_directory = os.path.join("data")
        meta_data_file = os.path.join(data_directory, 'form_daily_data.csv') 
        metadata = load_metadata(meta_data_file)

        #Data Cleaning & Transformation Pipeline
        cleaned_metadata = clean_metadata(metadata)
        standardized_metadata = standardize_metadata(cleaned_metadata)
        split_data = split_plant_numbers(standardized_metadata)
        adjusted_metadata = adjust_dates(split_data)

        #Metadata Validation
        issues = test_metadata(adjusted_metadata)

        #Fetch Data 
        output_parquet = os.path.join(data_directory, 'raw', 'daily_full_data.parquet')
        main(adjusted_metadata, AUTHORIZATION, output_parquet)

    except Exception as e:
        logging.critical(f"Critical error in the data pipeline: {e}")
