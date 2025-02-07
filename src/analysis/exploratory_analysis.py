from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import random
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots


## BASIC plot##
def plot_comparison(plant_uid, original_df, clean_df):
    """
    Plots various measurements for a specific plant, comparing original and clean data, with an additional 
    growth graph on a secondary axis.

    :param plant_uid: The ID of the plant to plot.
    :param original_df: DataFrame containing the original plant data.
    :param clean_df: DataFrame containing the cleaned plant data.
    """
    try:
        # Filter data for the specified plant ID
        original_data = original_df[original_df['unique_id'] == plant_uid]
        clean_data = clean_df[clean_df['unique_id'] == plant_uid]

        # Extract unique values for metadata
        plant_condition = clean_data['condition'].unique()
        plant_type = clean_data['plant_type'].unique()
        soil_type = clean_data['soil_sand'].unique()

        # Identify condition change time
        condition_change_time = clean_data[clean_data['condition'] != 'W'].first_valid_index()

        # Create subplots with a secondary y-axis for 'growth'
        fig = make_subplots(rows=6, cols=1, shared_xaxes=True, specs=[[{"secondary_y": True}] if i == 0 else [{}] for i in range(6)])

        # Plot Weight
        fig.add_trace(go.Scatter(x=original_data.index, y=original_data['s4'], mode='lines', name='Original Weight', line=dict(color='gray')), row=1, col=1)
        fig.add_trace(go.Scatter(x=clean_data.index, y=clean_data['s4'], mode='lines', name='Clean Weight', line=dict(color='blue')), row=1, col=1)
        
        # Plot PNW 
        pnw_original_data = original_data[['pnw']].dropna()
        fig.add_trace(go.Scatter(x=pnw_original_data.index, y=pnw_original_data['pnw'], mode='markers+lines', name='Plant net weight', line=dict(color='lightgreen', dash='dot')), row=1, col=1, secondary_y=True)
        pnw_clean_data = clean_data[['pnw']].dropna()
        fig.add_trace(go.Scatter(x=pnw_clean_data.index, y=pnw_clean_data['pnw'], mode='markers+lines', name='Plant net weight', line=dict(color='green')), row=1, col=1, secondary_y=True)

        # Plot Growth 
        if 'growth' in clean_data.columns: #80 so it would be at 4 in the morning
            clean_data['growth_line_weight'] = [clean_data['s4'].iloc[80] + clean_data['growth'].iloc[80] * (t.timestamp() - clean_data.index[80].timestamp()) for t in clean_data.index]
            if not clean_data['growth_line_weight'].isna().all():
                fig.add_trace(
                    go.Scatter(x=clean_data.index, y=clean_data['growth_line_weight'], mode='lines', name='Growth (Slope)', line=dict(dash='dash')),
                    row=1, col=1)
            

        # Plot Transpiration
        if 'transpiration' in clean_data.columns:
            #Transpiration between 4 AM and 8 PM - interval includes the start time but excludes the end time so it would be a round number of observations.
            fig.add_trace(go.Scatter(x=clean_data.between_time('04:00', '20:00', inclusive="left").index, y=clean_data['transpiration'].between_time('04:00', '20:00', inclusive="left"), mode='lines', name='Transpiration between 4and8', line=dict(color='blue')), row=2, col=1)
        
        elif 'dt' in clean_data.columns:
            # Filter non-NaN values for 'dt' before plotting
            dt_original = original_data[['dt']].dropna()
            dt_clean = clean_data[['dt']].dropna()
            
            fig.add_trace(go.Scatter(x=dt_original.index, y=dt_original['dt'], mode='markers+lines', name='Original Daily Transpiration', line=dict(color='lightgray')), row=2, col=1)
            fig.add_trace(go.Scatter(x=dt_clean.index, y=dt_clean['dt'], mode='markers+lines', name='Clean Daily Transpiration', line=dict(color='blue')), row=2, col=1)

        if 'normalized_transpiration' in clean_data.columns:
            fig.add_trace(go.Scatter(x=clean_data.index, y=clean_data['normalized_transpiration'], mode='lines', name='Normalized Transpiration', line=dict(color='purple', dash='dash')),
                          row=2, col=1)

        # Plot Light, Temperature, RH, and VPD
        fig.add_trace(go.Scatter(x=original_data.index, y=original_data['wspar'], mode='lines', name='Original Light', line=dict(color='orange', dash='dot')), row=3, col=1)
        fig.add_trace(go.Scatter(x=clean_data.index, y=clean_data['wspar'], mode='lines', name='Clean Light', line=dict(color='darkorange')), row=3, col=1)

        fig.add_trace(go.Scatter(x=original_data.index, y=original_data['wstemp'], mode='lines', name='Original Temperature', line=dict(color='red', dash='dot')), row=4, col=1)
        fig.add_trace(go.Scatter(x=clean_data.index, y=clean_data['wstemp'], mode='lines', name='Clean Temperature', line=dict(color='darkred')), row=4, col=1)

        fig.add_trace(go.Scatter(x=original_data.index, y=original_data['wsrh'], mode='lines', name='Original RH', line=dict(color='cyan', dash='dot')), row=5, col=1)
        fig.add_trace(go.Scatter(x=clean_data.index, y=clean_data['wsrh'], mode='lines', name='Clean RH', line=dict(color='darkcyan')), row=5, col=1)

        fig.add_trace(go.Scatter(x=original_data.index, y=original_data['vpd'], mode='lines', name='Original VPD', line=dict(color='magenta', dash='dot')), row=6, col=1)
        fig.add_trace(go.Scatter(x=clean_data.index, y=clean_data['vpd'], mode='lines', name='Clean VPD', line=dict(color='darkmagenta')), row=6, col=1)

        # Add vertical line for condition change time
        if condition_change_time is not None:
            for row in range(1, 7):
                fig.add_vline(x=condition_change_time, line_width=2, line_dash="dash", line_color="red", row=row, col=1)

        # Update layout
        fig.update_layout(
            height=1200,
            width=1500,
            title_text=f"Comparison of Original and Clean Data for Plant {plant_uid} (Condition: {plant_condition}, Crop: {plant_type}, Soil: {soil_type})",
            xaxis_title='Time',
        )
        fig.update_yaxes(title_text="Weight (g)", row=1, col=1)
        fig.update_yaxes(title_text="Growth", secondary_y=True, row=1, col=1)

        fig.show()
        print(f"Info: Comparison plot generated for plant ID: {plant_uid}")

    except Exception as e:
        print(f"Error in plot_comparison: {e}")


def plot_basic_plant_measurements(plant_uid, data, main_col='s4', col_p1='wspar', col_p2='wsrh', col_p3='wstemp'):
    """
    Plots various measurements for a specific plant using Plotly.

    :param plant_uid: The ID of the plant to plot.
    :param data: DataFrame containing the plant data.
    :param main_col: Column for main plot (default 's4').
    :param col_p1: Column for subplot 1 (default 'wspar').
    :param col_p2: Column for subplot 2 (default 'wsrh').
    :param col_p3: Column for subplot 3 (default 'wstemp').
    """
    try:
        # Define dynamic labels based on column names
        label_map = {
            's4': 'Weight',
            'pnw': 'Plant weight',
            'tr': 'Transpiration',
            'transpiration':'Transpiration (g/m)',
            'wspar': 'PAR Light',
            'wsrh': 'Relative Humidity (RH)',
            'vpd': 'Vapor Pressure Deficit (VPD)',
            'wstemp': 'Temperature'
        }

        # Dynamic labels fallback to the column name if not in label_map
        label_main = label_map.get(main_col, main_col)
        label_p1 = label_map.get(col_p1, col_p1)
        label_p2 = label_map.get(col_p2, col_p2)
        label_p3 = label_map.get(col_p3, col_p3)

        # Filter data for the specified plant ID
        plant_data = data[data['unique_id'] == plant_uid]
        
        # Finding the condition
        plant_condition = plant_data['condition'].unique()
        condition_change_time = plant_data[plant_data['condition'] != 'W'].first_valid_index()

        # Extract plant and soil type
        plant_type = plant_data['plant_type'].unique()[0]
        soil_type = plant_data['soil_sand'].unique()[0]

        # Create subplots layout
        fig = make_subplots(
            rows=2, cols=3,
            specs=[[{"colspan": 3}, None, None], [{}, {}, {}]],
            subplot_titles=(label_main, label_p1, label_p2, label_p3)
        )

        ## Main Measurement Plot ##
        if f'{main_col}_clean' in plant_data.columns:
            fig.add_trace(go.Scatter(x=plant_data.index, y=plant_data[f'{main_col}_clean'],
                                     mode='lines', name=f'{label_main} (clean)'), row=1, col=1)
        fig.add_trace(go.Scatter(x=plant_data.index, y=plant_data[main_col],
                                 mode='lines', name=label_main), row=1, col=1)

        # Mark condition change
        if condition_change_time is not None:
            fig.add_vline(x=condition_change_time, line_width=2, line_dash="dash", line_color="red", row=1, col=1)

        # Outliers
        if f'{main_col}_outlier' in plant_data.columns:
            outliers = plant_data[plant_data[f'{main_col}_outlier']]
            fig.add_trace(go.Scatter(x=outliers.index, y=outliers[f'{main_col}_outlier'],
                                     mode='markers', name=f'{label_main} Outliers', marker=dict(color='red', size=10)), row=1, col=1)
        
        # Plot the main column and its variants
        main_col_variants = [col for col in plant_data.columns if col.startswith(f"{main_col}_") and col != f"{main_col}_outlier"]
        # Plot each variant of the main column
        for variant in main_col_variants:
            variant_label = variant.replace(f"{main_col}_", "").replace("_", " ").capitalize()
            fig.add_trace(go.Scatter(x=plant_data.index, y=plant_data[variant],
                                     mode='lines', name=f'{label_main} ({variant_label})'), row=1, col=1)
            
        ## Subplots ##
        # Sub1
        if col_p1 in plant_data.columns:
            fig.add_trace(go.Scatter(x=plant_data.index, y=plant_data[col_p1],
                                     mode='lines', name=label_p1), row=2, col=1)
        # Sub2
        if col_p2 in plant_data.columns:
            fig.add_trace(go.Scatter(x=plant_data.index, y=plant_data[col_p2],
                                     mode='lines', name=label_p2), row=2, col=2)
        # Sub3
        if col_p3 in plant_data.columns:
            fig.add_trace(go.Scatter(x=plant_data.index, y=plant_data[col_p3],
                                     mode='lines', name=label_p3), row=2, col=3)

        # Final layout adjustments
        fig.update_layout(
            height=600, width=1000,
            title_text=f"Time Series Data for Plant {plant_uid} (Condition: {plant_condition[0]}, Crop: {plant_type}, Soil: {soil_type})"
        )
        fig.show()
        
        print(f"Info: Plot generated for plant ID: {plant_uid}")
    
    except Exception as e:
        print(f"Error in plot_basic_plant_measurements: {e}")


def plot_col(plant_uid, data, col_name):
    """
    Plots a specified column and its variants (e.g., cleaned, smoothed) for a given plant.

    :param plant_uid: The unique ID of the plant to plot.
    :param data: DataFrame containing the plant data.
    :param col_name: The column to plot.
    """
    try:
        # Define dynamic labels for known columns
        name_dict = {
            'wstemp': 'Temperature', 
            's4': 'Weight',
            'wsrh': 'Relative Humidity (RH)',
            'wspar': 'PAR Light',
            'tr': 'Transpiration',
            'transpiration': 'Transpiration'
        }

        # Use the provided label or fallback to the column name
        col_label = name_dict.get(col_name, col_name)

        # Filter data for the specified plant ID
        plant_data = data[data['unique_id'] == plant_uid]
        
        # Create the figure and plot the main column
        fig = go.Figure()
        if col_name in plant_data.columns:
            fig.add_trace(go.Scatter(x=plant_data.index, y=plant_data[col_name],
                                     mode='lines', name=col_label))
        else:
            print(f"Warning: Column '{col_name}' not found in the data.")
            return
        
        # Plot all variants of the column (excluding outliers)
        col_variants = [
            col for col in plant_data.columns 
            if col.startswith(f"{col_name}_") and col != f"{col_name}_outlier"
        ]
        
        for variant in col_variants:
            variant_label = variant.replace(f"{col_name}_", "").replace("_", " ").capitalize()
            fig.add_trace(go.Scatter(x=plant_data.index, y=plant_data[variant],
                                     mode='lines', name=f'{col_label} ({variant_label})'))

        # Check if 'outlier' column exists and plot outliers
        if f'{col_name}_outlier' in plant_data.columns:
            outliers = plant_data[plant_data[f'{col_name}_outlier']]
            fig.add_trace(go.Scatter(x=outliers.index, y=outliers[col_name],
                                     mode='markers', name=f'{col_label} Outliers',
                                     marker=dict(color='red', size=10)))

        # Update layout
        fig.update_layout(
            title=f"{col_label} Plot for Plant {plant_uid}",
            xaxis_title='Time',
            yaxis_title=col_label,
            height=500,
            width=800
        )
        fig.show()
        
        print(f"Plot generated for plant ID: {plant_uid}")
    
    except Exception as e:
        print(f"Error in plot_col: {e}")


## Condition

def plot_condition_pie_chart(df, condition_dict=None, colors=None):
    """
    Plots a pie chart showing the proportion of unique IDs by their dominant condition.
    Adds colored slices with black edges and positions labels next to the relevant pie pieces.

    Args:
        df (DataFrame): The DataFrame containing 'unique_id' and 'condition' columns.
        condition_dict (dict, optional): Mapping of condition codes to readable names.
        colors (list, optional): Custom list of color hex codes for the pie chart slices.

    Returns:
        None
    """
    try:
        # Assign each unique_id to its dominant condition (non-'W' if present, otherwise 'W')
        unique_condition = df.groupby('unique_id')['condition'].apply(
            lambda x: x[x != 'W'].iloc[0] if (x != 'W').any() else 'W'
        )

        # Apply condition mapping if provided
        if condition_dict:
            unique_condition = unique_condition.replace(condition_dict)

        # Count the conditions
        unique_condition_counts = unique_condition.value_counts()

        labels = unique_condition_counts.index
        sizes = unique_condition_counts.values

        # Default color palette if none provided
        default_colors = ['#f9c74f', '#90be6d', '#43aa8b', '#577590', '#f94144', '#f3722c', '#f8961e']
        colors = colors if colors else default_colors

        # Plot pie chart
        plt.figure(figsize=(8, 8))
        wedges, texts, autotexts = plt.pie(
            sizes, labels=None, colors=colors[:len(sizes)], startangle=180, 
            autopct='%1.1f%%', wedgeprops={'edgecolor': 'black'}, pctdistance=0.80
        )

        # Adjust label positions
        for i, a in enumerate(autotexts):
            a.set_text(f'{sizes[i]} ({a.get_text()})')
            a.set_color('black')
            a.set_fontsize(10)
        
        # Add labels outside the pie
        for i, text in enumerate(texts):
            text.set_text(labels[i])
            text.set_fontsize(12)
        
        plt.title('Proportion of Unique IDs by Condition')
        plt.axis('equal')
        plt.tight_layout()  # Prevent overlapping
        plt.show()

    except Exception as e:
        print(f"Error generating pie chart: {e}")