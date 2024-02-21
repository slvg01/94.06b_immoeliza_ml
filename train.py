import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # outliers
    num_cols =[
    "total_area_sqm", 
    "surface_land_sqm",
    "nbr_frontages",
    "nbr_bedrooms",
    "terrace_sqm",
    #"garden_sqm",
    "primary_energy_consumption_sqm",
    "cadastral_income",
    ]
    
    print(len(data))
    for col in num_cols : 
        Q75 = data[col].quantile(0.75)
        Q25 = data[col].quantile(0.25)
        IQR = Q75-Q25
        upper = Q75 + 3.5 * IQR
        lower = Q25 - 3.5 * IQR
        data = data[((data[col] < upper) & (data[col] > lower)) | (data[col].isnull()) | (data[col] == 0)]
    print(len(data))

  
    # Standardize numerical data 
    num_cols2 = [
    "latitude",
    "longitude",
    "construction_year",
    "total_area_sqm", 
    "surface_land_sqm",
    "nbr_frontages",
    "nbr_bedrooms",
    "terrace_sqm",
    "garden_sqm",
    "primary_energy_consumption_sqm",
    "cadastral_income",    
    ]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[num_cols2])
    scaled_data = pd.DataFrame(scaled_data, columns=num_cols2)
    scaled_data.index = data.index
    data[num_cols2] = scaled_data
    



    # Define features to use
    num_features = [
        "construction_year",
        "latitude",
        "longitude",
        "total_area_sqm",
        "surface_land_sqm",
        "nbr_frontages",
        "nbr_bedrooms",
        "terrace_sqm",
        "primary_energy_consumption_sqm",
        "cadastral_income",
        "garden_sqm"
    ]
    fl_features = [
        "fl_terrace",
        "fl_open_fire",
        "fl_swimming_pool",
        "fl_garden",
        "fl_double_glazing",
        
        
    ]
    cat_features = [
        "subproperty_type",
        "locality",
        "equipped_kitchen",
        "state_building",
        "epc",
        
    ]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
        [
            X_train[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    X_test = pd.concat(
        [
            X_test[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    #(f"Features: \n {X_train.columns.tolist()}")

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = r2_score(y_train, model.predict(X_train))
    test_score = r2_score(y_test, model.predict(X_test))
    print(f"Train R² score: {train_score}")
    print(f"Test R² score: {test_score}")

    # Save the model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": model,
    }
    joblib.dump(artifacts, "models/Linear_artifacts.joblib")

    

if __name__ == "__main__":
    train()

