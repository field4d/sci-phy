## go to https://www.scopus.com/search/form.uri?display=basic#basic
# search for ("transpiration" OR "evapotranspiration" OR "Et" ) AND ( "machine learning" OR "Artificial Intelligence" ) in title abstract or keywords
# filter : Between 2010 and 2024. type: Article. Subject area : Environmental Science , Earth and Planetary Sciences , Agricultural and Biological Sciences.
# download csv of 1-2061 documents , with "Citation information" and "Abstract & keywords" 

import pandas as pd

# Load the uploaded file
file_path = "G:\My Drive\Shani Friedman\HUJI\ML\Review\scopus.csv"
df = pd.read_csv(file_path)

# Define keywords related to ML application in transpiration/evapotranspiration prediction
ml_keywords = ["machine learning", "artificial intelligence", "deep learning", "neural network", 
               "random forest", "support vector machine", "svm", "gradient boosting", "lstm", "cnn"]
transpiration_keywords = ["transpiration", "evapotranspiration"]
prediction_keywords = ["predict", "prediction", "forecast", "estimating","estimates", "modeling"]

# Irrelevant topics to exclude
irrelevant_keywords = ["groundwater", "aquifer", "subsurface", "streamflow", 
                       "ungauged basins", "heavy metals", "soil pollution", 
                       "suspended sediment concentration", "ssc", "erosion", 
                       "crop model", "yield forecast",
                       "drought forecasting", "standardized precipitation evapotranspiration index",
                       "Gross primary productivity",
                       "fire", "burn rate"] 


# Define measurement methods categories
new_grouped_methods = {
    "eddy covariance": ["eddy covariance", "flux tower"],
    "weather station": ["weather station", "synoptic station", "meteorological", "meteorology", "environmental sensor", "environmental data"],
    "remote sensing": ["remote sensing", "satellite", "thermal", "multispectral", "camera", "imaging", "image", "modis"],
    "load cells": ["lysimeter", "load cells", "gravimetric"],
    "sap flow": ["sap flow", "sap flux", "xylem flow"],
    "field measurement": ["ground-based", "field measurement", "porometer"],
    "phenotyping": ["phenotyping"],
    "soil water balance": ["soil moisture sensors", "neutron probe", "tdr", "soil water depletion"],
    #"atmospheric budget": ["aerodynamic method", "bowen ratio", "energy balance"],
    "empirical models": ["fao", "penman-monteith","penman","monteith", "priestley-taylor", "hargreaves", "blaney-criddle"],
    "stomatal conductance": ["gas exchange", "infrared gas analyzer", "li-cor", "photosynthesis system"],
    "psychrometry": ["psychrometer", "dew point method"]
    #"electronic nose": ["electronic nose"] 
}


new_grouped_features = {
    "climate & meteorology": [
        "temperature", "humidity", "solar radiation", "wind speed", "vapor pressure deficit", 
        "vpd", "precipitation", "climate", "penman-monteith","penman","monteith", "fao", "evapotranspiration", 
        "air pressure", "dew point", "atmospheric water content", "net radiation"
    ],
    "vegetation indices": [
        "normalized difference vegetation index", "leaf area index", "plant coverage", 
        "ndvi", "lai", "sif", "vegetation indices", "chlorophyll index", "green ndvi", 
        "enhanced vegetation index", "evi", "water index", "wvi", "red edge index"
    ],
    "soil": [
        "soil moisture", "available water", "soil temperature", "soil water potential", "soil texture", 
        "soil electrical conductivity", "soil evaporation", "soil heat flux"
    ],
    "plant physiology": [
        "stomata", "physiolog", # physiology, physiologic
        "plant weight", "plant height", "leaf number","plant growth"
        "gas exchange", "photosynthesis",
        "transpiration rate", "water use efficiency", "sap flow", "xylem conductance", 
        "stomatal conductance", "leaf water potential", "osmotic potential", "relative water content"
    ] #volatile compounds
}


# Function to classify articles
def classify_article(abstract):
    if pd.isna(abstract):
        return {"ML_ET_Related": "No", "ET_Prediction": "No"}
    
    abstract_lower = abstract.lower()
    ml_match = any(kw in abstract_lower for kw in ml_keywords)
    transpiration_match = any(kw in abstract_lower for kw in transpiration_keywords)
    prediction_match = any(kw in abstract_lower for kw in prediction_keywords)
    irrelevant_match = any(kw in abstract_lower for kw in irrelevant_keywords)

    return {
        "ML_ET_Related": "Yes" if ml_match and transpiration_match else "No",
        "ET_Prediction": "Yes" if ml_match and transpiration_match and prediction_match and not irrelevant_match else "No"
    }

# Function to extract measurement methods and features
def extract_measurement_info(abstract):
    if pd.isna(abstract):
        return {"Measurement Methods": "Not Specified", "Features Used": "Not Specified"}
    
    abstract_lower = abstract.lower()
    measurement_categories = [category for category, keywords in new_grouped_methods.items() if any(kw in abstract_lower for kw in keywords)]
    feature_categories = [category for category, keywords in new_grouped_features.items() if any(kw in abstract_lower for kw in keywords)]
    
    return {
        "Measurement Methods": ", ".join(measurement_categories) if measurement_categories else "Not Specified",
        "Features Used": ", ".join(feature_categories) if feature_categories else "Not Specified",
    }

# Apply classification and extraction functions
classification_results = df["Abstract"].apply(classify_article)
measurement_info = df["Abstract"].apply(extract_measurement_info)

# Convert extracted data into separate columns
df["ML_ET_Related"] = classification_results.apply(lambda x: x["ML_ET_Related"])
df["ET_Prediction"] = classification_results.apply(lambda x: x["ET_Prediction"])
df["Measurement Methods"] = measurement_info.apply(lambda x: x["Measurement Methods"])
df["Features Used"] = measurement_info.apply(lambda x: x["Features Used"])

# Function to extract binary flags for measurement methods
def get_measurement_flags(abstract):
    if pd.isna(abstract):
        return {method: 0 for method in new_grouped_methods.keys()}
    
    abstract_lower = abstract.lower()
    method_flags = {category: 1 if any(kw in abstract_lower for kw in keywords) else 0 for category, keywords in new_grouped_methods.items()}
    
    return method_flags

# Function to extract binary flags for features
def get_feature_flags(abstract):
    if pd.isna(abstract):
        return {feature: 0 for feature in new_grouped_features.keys()}
    
    abstract_lower = abstract.lower()
    feature_flags = {category: 1 if any(kw in abstract_lower for kw in keywords) else 0 for category, keywords in new_grouped_features.items()}
    
    return feature_flags

# Create DataFrame for measurement methods and features
measurement_flags = df["Abstract"].apply(get_measurement_flags).apply(pd.Series)
feature_flags = df["Abstract"].apply(get_feature_flags).apply(pd.Series)


# Create MultiIndex columns
measurement_index = pd.MultiIndex.from_tuples(
    [("Methods", method) for method in measurement_flags.columns],
    names=["Category", "Measurement Method"]
)
feature_index = pd.MultiIndex.from_tuples(
    [("Features", feature) for feature in feature_flags.columns],
    names=["Category", "Feature Type"]
)
# Assign MultiIndex to DataFrames
measurement_flags.columns = measurement_index
feature_flags.columns = feature_index


selected_columns = ["Authors", "Title", "Year", "DOI", "Abstract", 
                        "Author Keywords", "Index Keywords", 
                        "ML_ET_Related", "ET_Prediction", 
                        "Measurement Methods", "Features Used"]

# Filter main DataFrame
df_filtered = df.loc[df["ET_Prediction"] == "Yes", selected_columns]

# Merge with MultiIndex DataFrames
# Ensure measurement_flags and feature_flags match df_selected's index
measurement_flags = measurement_flags.loc[df_filtered.index]
feature_flags = feature_flags.loc[df_filtered.index]

df_final = pd.concat([df_filtered, measurement_flags, feature_flags], axis=1)


# Save full processed dataset
df.to_csv(r"G:\My Drive\Shani Friedman\HUJI\ML\Review\annotated_scopus.csv")

# Save filtered dataset with MultiIndex
df_final.to_csv(r"G:\My Drive\Shani Friedman\HUJI\ML\Review\annotated_scopus_filtered.csv")
