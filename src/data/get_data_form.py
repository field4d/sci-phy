# Importing necessary libraries
import datetime
import pandas as pd
import os
import numpy as np
import re

# Function to validate Experiment ID
def validate_experiment_id(exp_id):
    """
    Validate that the Experiment ID contains numbers only.

    Args:
        exp_id (str): The Experiment ID input by the user.

    Returns:
        str: Validated Experiment ID.

    Raises:
        ValueError: If the input is not numeric.
    """
    if exp_id.isdigit():
        return exp_id
    raise ValueError("Experiment ID must contain numbers only.")

# Function to validate Plant ID(s)
def validate_plant_ids(plant_ids):
    """
    Validate that Plant IDs contain numbers or numbers with spaces.

    Args:
        plant_ids (str): The Plant IDs input by the user.

    Returns:
        str: Validated Plant IDs.

    Raises:
        ValueError: If the input does not match the expected pattern.
    """
    if re.fullmatch(r'(\d+\s?)+', plant_ids):
        return plant_ids
    raise ValueError("Plant IDs must be numbers or numbers with spaces.")

# Function to get validated user input
def get_input(prompt, explanation="", required=True, options=None, validation_func=None):
    """
    Get validated user input from the command line with optional validation and predefined options.

    Args:
        prompt (str): The message displayed to the user.
        explanation (str): Additional context for the input.
        required (bool): Indicates if the input is mandatory.
        options (list, optional): List of valid options.
        validation_func (function, optional): Function to validate the input.

    Returns:
        str: Validated user input.
    """
    print("\n----\n")
    print(f"\033[1m\033[4m{prompt}\033[0m")
    if explanation:
        print(f"Note: {explanation}")

    # Handling selection from predefined options
    if options:
        for idx, option in enumerate(options, start=1):
            print(f"  {idx}. {option}")

        while True:
            choice = input("Select an option (number): ")
            if choice.isdigit() and 1 <= int(choice) <= len(options):
                selected_option = options[int(choice) - 1]
                return input("Enter details: ") if selected_option.lower() == "other" else selected_option
            print("Invalid choice. Please select a valid number.")
    else:
        # Handling free-text input with optional validation
        while True:
            user_input = input("Your answer: ")
            if not user_input and required:
                print("This field is required.")
                continue
            try:
                return validation_func(user_input) if validation_func else user_input
            except ValueError as e:
                print(e)

# Function to get validated date input
def get_date_input(prompt, explanation=""):
    """
    Get a validated date input from the user in YYYY-MM-DD format.

    Args:
        prompt (str): The message displayed to the user.
        explanation (str): Additional context for the input.

    Returns:
        datetime.date or np.nan: Parsed date if valid, otherwise NaN.
    """
    print("\n----\n")
    print(f"Note: {explanation}")
    while True:
        date_input = input(f"\033[1m\033[4m{prompt}\033[0m (YYYY-MM-DD): ")
        try:
            return datetime.datetime.strptime(date_input, "%Y-%m-%d").date() if date_input else np.nan
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD.")

# Function to collect user inputs
def collect_form_data():
    """
    Collects form data interactively from the user.

    Returns:
        dict: A dictionary containing the collected form data.
    """
    data = {}

    data['Greenhouse'] = get_input("Select Greenhouse", "Choose the greenhouse.", options=["Icore","Lizzy", "ISURF", "Room101", "Moris"])
    data['Experiment ID'] = get_input("Enter Experiment ID", "Numbers only.", validation_func=validate_experiment_id)
    data['Plant ID(s)'] = get_input("Enter Plant ID(s)", "Numbers or numbers with spaces.", validation_func=validate_plant_ids)
    data['Keep'] = get_input("Is this plant good for experiments?", options=["Yes", "Yes, some dates", "No"])
    data['Treatment'] = get_input("Which treatment did this plant get?", options=["Control", "Cutting", "Drought - Terminal", "Drought - With some irrigation", "Salt", "Fol infection", "Not relevant"])
    data['Start Date'] = get_date_input("Enter Start Date", "The day from which data collection should start.")
    data['Treatment Start Date'] = get_date_input("Enter Treatment Start Date", "The first day the plant is stressed") if data['Treatment'].lower() != 'control' else np.nan
    data['End Date'] = get_date_input("Enter End Date", "End of data OR The date the recovery started (for example : the irrigation started at 20:00 at yyyy-mm-dd)")
    data['Remove Dates'] = get_input("Remove some dates?", "Select if specific dates should be excluded.", options=["No", "Specific dates", "Span from remove1 to remove2"])
    if data['Remove Dates'] != "No":
        data['Remove Date 1'] = get_date_input("Enter Remove Date 1", "The day to remove or Start date of the range to be excluded.")
        data['Remove Date 2'] = get_date_input("Enter Remove Date 2", "The day to remove or End date of the range to be excluded.")
    else:
        data['Remove Date 1'], data['Remove Date 2'] = np.nan, np.nan

    data['Crop Type'] = get_input("Select Crop Type", "Choose the type of crop.", options=["Tomato", "Rice", "Wheat", "Barley", "Maize", "Melon", "Pepper", "Potato", "Arabidopsis", "Cereal", "Bayam", "Kailan", "Lettuce", "Other"])
    data['Crop Name'] = get_input("Enter Crop Name", "Provide the specific line name of the crop. (like M82 or Svevo)")
    data['Unseen by User?'] = get_input("Was this plant marked as unseen by user?", "Indicate if the plant was marked 'bad' by a user.", options=["Yes", "No", "I don't know"])
    data['Daily Check'] = get_input("Did you Check the daily transpiration?", "Is the plant not in \033[4mpot limitation\033[0m? does this plant act as expected?", options=["Yes", "No"])
    data['Info'] = get_input("more info?" )

    return data

# Function to review and edit data with date range validation
def review_and_edit_data(data):
    """
    Review and allow editing of collected data. Ensures date ranges do not exceed 100 days.

    Args:
        data (dict): The form data to review and edit.

    Returns:
        dict: The reviewed and possibly modified form data.
    """
    
    # Review and edit all data
    print("\n----\n")
    print("\n\033[1mReview Collected Data:\033[0m") # bold print
    for key, value in data.items():
        print(f"\033[4m{key}\033[0m: {value}") # underline print

    ## User warning
    # Warning if date ranges exceed 100 days
    date_diff = (data['End Date'] - data['Start Date']).days
    treatment_diff = (data['End Date'] - data['Treatment Start Date']).days if data['Treatment Start Date'] is not np.nan else 0
    if date_diff > 100 or treatment_diff > 100:
        print("⚠️!!! WARNING!!!⚠️")
        print("WARNING: The date range exceeds 100 days. Please review the dates carefully and adjust if necessary")

    # Validate Treatment Start AND Remove Dates are within Start and End Dates
    if not pd.isna(data.get('Treatment Start')) and (data['Treatment Start'] < data['Start Date'] or data['Treatment Start'] > data['End Date']):
        print("⚠️!!! WARNING!!! ⚠️")
        print("WARNING: 'Treatment Start' is outside the range of 'Start Date' and 'End Date'. Please review and adjust.")

    if not pd.isna(data.get('Remove Date 1')) and (data['Remove Date 1'] < data['Start Date'] or data['Remove Date 1'] > data['End Date']):
        print("⚠️!!! WARNING!!!⚠️")
        print("WARNING: 'Remove Date 1' is outside the range of 'Start Date' and 'End Date'. Please review and adjust.")

    if not pd.isna(data.get('Remove Date 2')) and (data['Remove Date 2'] < data['Start Date'] or data['Remove Date 2'] > data['End Date']):
        print("⚠️!!! WARNING!!!⚠️")
        print("WARNING: 'Remove Date 2' is outside the range of 'Start Date' and 'End Date'. Please review and adjust.")

    while True:
        edit_choice = get_input("Would you like to edit any field?", options=["Yes", "No"])
        if edit_choice == "No":
            break

        field_to_edit = get_input("Enter the field name you want to edit:")
        if field_to_edit in data:
            if "Date" in field_to_edit or "Start" in field_to_edit:
                data[field_to_edit] = get_date_input(f"Re-enter {field_to_edit}", "Please use YYYY-MM-DD.")
            else:
                data[field_to_edit] = get_input(f"Re-enter {field_to_edit}", "Add full answer copied from above")
        else:
            print("Invalid field name. Please try again.")

    return data

# Function to save data to a CSV file
def save_data_to_file(data, file_name="form_data.csv"):
    """
    Save the collected form data to a CSV file.

    Args:
        data (dict): The form data to save.
        file_name (str): The name of the file to save to.
    """
    try:
        directory = os.path.join("data")
        os.makedirs(directory, exist_ok=True)
        file_path = os.path.join(directory, file_name)

        df = pd.DataFrame([data])
        if os.path.exists(file_path):
            existing_df = pd.read_csv(file_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(file_path, index=False)
        print(f"\n\033[1m\033[92m✅ Data successfully saved to {file_path} ✅\033[0m")
    except Exception as e:
        print(f"Failed to save data: {e}")


# Main program execution
if __name__ == "__main__":
    """
    Main execution block to collect form data, display the results, and save to a file.
    """
    try:
        while True:
            form_data = collect_form_data()  # Collect form data from the user
            form_data = review_and_edit_data(form_data)  # Review and validate data
            save_data_to_file(form_data)  # Save data to file

            repeat = get_input("Would you like to fill another form?", options=["Yes", "No"])
            if repeat == "No":
                break
    except KeyboardInterrupt:
        print("\nProcess interrupted by the user.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

