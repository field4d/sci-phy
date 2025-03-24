# Model Testing App 🚀

## Overview

This Streamlit-based web application allows users to test pre-trained machine learning models on plant transpiration data. It enables users to input their experimental data and visualize model predictions.

## Features

✅ Load and test models on user-provided data\
✅ Select plant type and experiment details\
✅ Retrieve and visualize transpiration data\
✅ Generate model evaluation metrics and plots

## How to Use

1. **Enter Details**: Provide the Control ID, Experiment ID, and Plant ID.
2. **Select Dates**: Choose a date range for analysis.
3. **Pick Plant Type**: Tomato or Cereal.
4. **Check Irrigation Status**: Mark whether the plant was well irrigated.
5. **Retrieve Data**: Click **"Test Model"** to load data.
6. **Evaluate Model**: If the data looks correct, click **"This is my data! Let's test the Model"** to run the models and visualize results.

## Setup Instructions

### **1. Clone the Repository**

```sh
git clone https://github.com/yourusername/ModelTestingApp.git
cd ModelTestingApp
```

### **2. Set Up the Environment**

Ensure you have Python installed. Create a virtual environment and install dependencies:

```sh
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
pip install -r requirements.txt
```

### **3. Run the App**

Start the Streamlit application with:

```sh
streamlit run app/TestModelApp.py
```

## Deployment (Optional)

You can deploy this app using **Streamlit Cloud**:

1. Push the repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Connect your repository.
4. Deploy the app.

## Repository Structure

```
/ModelTestingApp
│── app/                     # Contains main app-related scripts
│   ├─ TestModelApp.py       # Main Streamlit app file
│   └─ app_functions.py      # Contains utility functions (e.g., data retrieval & model testing)
│─ models/                   # Stores trained models (not included in the public repo)
│─ data/                     # Example dataset for users to test
│─ requirements.txt          # List of dependencies for setting up the environment
│─ README.md                 # Documentation explaining the app, setup, and usage
│─ .gitignore                # Excludes unnecessary files from version control
│─ LICENSE                   # Open-source license (optional)
```

## Contributing

Contributions are welcome! Feel free to fork the repo, make improvements, and submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

## Contact

For any issues or suggestions, feel free to reach out!

---

