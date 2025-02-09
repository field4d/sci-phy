import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def normalize_to_growth(df, transpiration_col='transpiration', growth_col='growth'):
    """
    Normalize transpiration by the 'growth' column.

    Args:
        df (pd.DataFrame): DataFrame containing transpiration and growth data.
        transpiration_col (str): Name of the transpiration column.
        growth_col (str): Name of the growth column.

    Returns:
        pd.DataFrame: DataFrame with a new column 'transpiration_growth_normalized'.
    """
    try:
        logging.info("Normalizing transpiration to growth.")
        df['transpiration_growth_normalized'] = df[transpiration_col] / df[growth_col]
        logging.info("Normalization to growth completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error normalizing to growth: {e}")
        raise

def normalize_to_pnw(df, transpiration_col='transpiration', pnw_col='pnw'):
    """
    Normalize transpiration by the 'pnw' (Plant Net Weight) column.

    Args:
        df (pd.DataFrame): DataFrame containing transpiration and pnw data.
        transpiration_col (str): Name of the transpiration column.
        pnw_col (str): Name of the pnw column.

    Returns:
        pd.DataFrame: DataFrame with a new column 'transpiration_pnw_normalized'.
    """
    try:
        logging.info("Normalizing transpiration to pnw.")
        df['transpiration_pnw_normalized'] = df[transpiration_col] / df[pnw_col]
        logging.info("Normalization to pnw completed successfully.")
        return df
    except Exception as e:
        logging.error(f"Error normalizing to pnw: {e}")
        raise


def normalize_to_daily_95th_percentile(df, transpiration_col='transpiration', unique_id_col='unique_id'):
    """
    Normalize transpiration using the daily 95th percentile for each unique ID.

    Args:
        df (pd.DataFrame): DataFrame containing transpiration data.
        transpiration_col (str): Name of the transpiration column.
        unique_id_col (str): Name of the unique identifier column.

    Returns:
        pd.DataFrame: DataFrame with a new column 'transpiration_daily_95_normalized'.
    """
    try:
        logging.info("Normalizing transpiration to daily 95th percentile.")
        df.index = pd.to_datetime(df.index)

        # Filter data between 4 AM and 8 PM for percentile calculation
        filtered_df = df.between_time('04:00', '20:00').copy()

        # Calculate the 95th percentile per day and unique_id
        percentiles = filtered_df.groupby(
            [filtered_df[unique_id_col], pd.Grouper(freq='D')]
        )[transpiration_col].transform(lambda x: np.percentile(x, 95))

        # Assign these percentiles back to the filtered_df
        filtered_df.loc[:, 'daily_95th'] = percentiles
        
        # Reset index to merge on both index and unique_id
        df_reset = df.reset_index()
        filtered_df_reset = filtered_df.reset_index()

        # Merge on both timestamp (index) and unique_id
        df_merged = df_reset.merge(
            filtered_df_reset[[ unique_id_col, 'timestamp', 'daily_95th']],
            on=['timestamp', unique_id_col],
            how='left'
        )

        # Set the index back to datetime after merge
        df_merged.set_index('timestamp', inplace=True)

        # Fill NaNs for times outside 4 AM - 8 PM window
        df_merged['daily_95th'] = df_merged.groupby([df_merged[unique_id_col], pd.Grouper(freq='D')])['daily_95th'].transform(lambda x: x.ffill().bfill())

        # Normalize transpiration
        df_merged['transpiration_daily_95_normalized'] = df_merged[transpiration_col] / df_merged['daily_95th']

        # Drop the temporary 'daily_95th' column
        df_merged.drop(columns=['daily_95th'], inplace=True)

        logging.info("Normalization to daily 95th percentile completed successfully.")
        return df_merged

    except Exception as e:
        logging.error(f"Error normalizing to daily 95th percentile: {e}")
        raise


def normalize_to_similar_treatments(df, transpiration_col='transpiration', condition_col='condition',unique_id_col = 'unique_id', exp_id_col='exp_ID', control_id_col='gh_ID', soil_type_col='soil_sand', plant_type_col='plant_type'):
    """
    Normalize transpiration using the mean and 95th percentile of similar treatments on the same day.
    Similar treatments are defined by matching condition, experiment ID, control ID, soil type, and plant type.

    Args:
        df (pd.DataFrame): DataFrame containing transpiration data.
        transpiration_col (str): Name of the transpiration column.
        condition_col (str): Name of the condition column.
        unique_id_col (str): Name of the unique identifier column.
        exp_id_col (str): Name of the experiment ID column.
        control_id_col (str): Name of the control ID column.
        soil_type_col (str): Name of the soil type column.
        plant_type_col (str): Name of the plant type column.

    Returns:
        pd.DataFrame: DataFrame with new columns 'transpiration_similar_mean_normalized' and 'transpiration_similar_95_normalized'.
    """
    try:
        logging.info("Normalizing transpiration to similar treatments.")
        df.index = pd.to_datetime(df.index)
        filtered_df = df.between_time('04:00', '20:00').copy()

        group_cols = [condition_col, exp_id_col, control_id_col, soil_type_col, plant_type_col, pd.Grouper(freq='D')]

        filtered_df.loc[:, 'similar_treatment_mean'] = filtered_df.groupby(group_cols)[transpiration_col].transform('mean')
        filtered_df.loc[:, 'similar_treatment_95th'] = filtered_df.groupby(group_cols)[transpiration_col].transform(lambda x: np.percentile(x, 95))

        df_reset = df.reset_index()
        filtered_df_reset = filtered_df.reset_index()

        df_merged = df_reset.merge(
            filtered_df_reset[[unique_id_col, 'timestamp', 'similar_treatment_mean', 'similar_treatment_95th']],
            on=['timestamp', unique_id_col],
            how='left'
        )

        df_merged.set_index('timestamp', inplace=True)
        df_merged['similar_treatment_mean'] = df_merged.groupby(group_cols[:-1])['similar_treatment_mean'].transform(lambda x: x.ffill().bfill())
        df_merged['similar_treatment_95th'] = df_merged.groupby(group_cols[:-1])['similar_treatment_95th'].transform(lambda x: x.ffill().bfill())

        df_merged['transpiration_similar_mean_normalized'] = df_merged[transpiration_col] / df_merged['similar_treatment_mean']
        df_merged['transpiration_similar_95_normalized'] = df_merged[transpiration_col] / df_merged['similar_treatment_95th']

        df_merged.drop(columns=['similar_treatment_mean', 'similar_treatment_95th'], inplace=True)

        logging.info("Normalization to similar treatments completed successfully.")
        return df_merged

    except Exception as e:
        logging.error(f"Error normalizing to similar treatments: {e}")
        raise

def apply_all_normalizations(df):
    """
    Apply all transpiration normalization functions to the dataset.

    Args:
        df (pd.DataFrame): The DataFrame containing transpiration data.

    Returns:
        pd.DataFrame: DataFrame with all normalized transpiration columns added.
    """
    try:
        logging.info("Starting all normalization processes.")
        df = normalize_to_growth(df)
        df = normalize_to_pnw(df)
        df = normalize_to_daily_95th_percentile(df)
        df = normalize_to_similar_treatments(df)
        logging.info("All normalization processes completed successfully.")
        return df
    except Exception as e:
        logging.critical(f"Critical error in normalization pipeline: {e}")
        raise
    