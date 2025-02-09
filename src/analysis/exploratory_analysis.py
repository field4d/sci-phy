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
import plotly.graph_objects as go
import plotly.express as px


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
            clean_data['growth_line_weight'] = [clean_data['s4'].iloc[80] + clean_data['growth'].iloc[80] * ((t.timestamp() - clean_data.index[80].timestamp())/60) for t in clean_data.index]
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

        # Plot all variants of the transpiration column
        col_variants = [col for col in clean_data.columns if col.startswith(f"transpiration_")]
        for variant in col_variants:
            variant_label = variant.replace(f"transpiration_", "").replace("_", " ").capitalize()
            fig.add_trace(go.Scatter(x=clean_data.index, y=clean_data[variant],
                                     mode='lines', name=f'Transpiration ({variant_label})'), row=2, col=1)
            
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
            'transpiration':'Transpiration',
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

## Plot transpiration normalization aggregated by condition

def plot_transpiration_by_condition(df, gh_id, exp_id, normalized_col=None, time_col='timestamp', condition_col='condition'):
    """
    This function filters the input DataFrame based on the provided greenhouse ID (gh_id) and experiment ID (exp_id),
    then groups the data by timestamp and condition to calculate the mean and standard deviation of the transpiration
    data. It generates a plot with lines representing the mean transpiration and shaded areas representing the standard
    deviation, which can be toggled on and off.

    Parameters:
        df (pd.DataFrame): The original DataFrame containing transpiration data.
        gh_id (int): Greenhouse ID to filter the data.
        exp_id (int): Experiment ID to filter the data.
        normalized_col (str, optional): The column name for normalized transpiration data. If None, plots raw transpiration.
        time_col (str, optional): The column name for timestamps.
        condition_col (str, optional): The column name for condition labels.
    """

    # Filter data based on gh_id and exp_id
    filtered_df = df[(df['gh_ID'] == gh_id) & (df['exp_ID'] == exp_id)]

    # Check if filtered_df is empty
    if filtered_df.empty:
        print(f'Are you sure you chose the correct exp_id and greenhouse id? gh {gh_id} exp {exp_id}')
        return

    # Determine columns to aggregate
    agg_columns = ['transpiration']
    if normalized_col:
        agg_columns.append(normalized_col)

    # Grouping by timestamp and condition to calculate mean and std
    mean_std_df = filtered_df.groupby([filtered_df.index, condition_col])[agg_columns].agg(['mean', 'std']).reset_index()

    # Flatten the MultiIndex
    mean_std_df.columns = ['_'.join(col).strip() if col[1] else col[0] for col in mean_std_df.columns]

    # Convert relevant columns to numeric, handling errors gracefully
    numeric_columns = [col for col in mean_std_df.columns if any(x in col for x in ['mean', 'std'])]
    mean_std_df[numeric_columns] = mean_std_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

    # Convert index to datetime
    mean_std_df[time_col] = pd.to_datetime(mean_std_df[time_col], errors='coerce')

    # Drop rows with NaN values in key columns
    required_columns = [time_col, 'transpiration_mean', 'transpiration_std']
    if normalized_col:
        required_columns += [f'{normalized_col}_mean', f'{normalized_col}_std']

    agg_df_clean = mean_std_df.dropna(subset=required_columns)

    # Determine plot titles outside the loop to avoid UnboundLocalError
    if normalized_col:
        title = 'Normalized Transpiration with Toggleable Standard Deviation'
        yaxis_title = 'Normalized Transpiration'
    else:
        title = 'Transpiration with Toggleable Standard Deviation'
        yaxis_title = 'Transpiration'

    # Generate a color palette with enough distinct colors
    color_palette = px.colors.qualitative.Plotly + px.colors.qualitative.Dark24 + px.colors.qualitative.Alphabet

    # Plotting
    fig = go.Figure()

    for idx, condition in enumerate(agg_df_clean[condition_col].unique()):
        cond_data = agg_df_clean[agg_df_clean[condition_col] == condition]

        color = color_palette[idx % len(color_palette)]

        if normalized_col:
            mean_col = f'{normalized_col}_mean'
            std_col = f'{normalized_col}_std'
            line_name = f'Normalized - {condition}'
            std_name = f'STD (Normalized) - {condition}'
            fillcolor = color.replace('1)', '0.2)').replace('rgb', 'rgba')
        else:
            mean_col = 'transpiration_mean'
            std_col = 'transpiration_std'
            line_name = f'Transpiration - {condition}'
            std_name = f'STD - {condition}'
            fillcolor = color.replace('1)', '0.2)').replace('rgb', 'rgba')

        # Add mean line
        fig.add_trace(go.Scatter(
            x=cond_data[time_col],
            y=cond_data[mean_col],
            mode='lines',
            name=line_name,
            line=dict(width=2, color=color)
        ))

        # Add shaded area for std deviation (toggleable)
        fig.add_trace(go.Scatter(
            x=pd.concat([cond_data[time_col], cond_data[time_col][::-1]]),
            y=pd.concat([
                (cond_data[mean_col] + cond_data[std_col]),
                (cond_data[mean_col] - cond_data[std_col])[::-1]
            ]),
            fill='toself',
            fillcolor=fillcolor,
            line=dict(color='rgba(255,255,255,0)'),
            name=std_name,
            showlegend=True,
            opacity=0.4
        ))

    # Layout for Transpiration
    fig.update_layout(
        title=title,
        xaxis_title='Timestamp',
        yaxis_title=yaxis_title,
        legend_title='Condition',
        template='plotly_white'
    )

    fig.show()
