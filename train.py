import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV



def ouliers(dataset):

    # outliers
    cols_outliers = [
        "total_area_sqm",
        "surface_land_sqm",
        "nbr_frontages",
        "nbr_bedrooms",
        "terrace_sqm",
        "primary_energy_consumption_sqm",
        "cadastral_income",
    ]
    
    print(len(dataset))
    for col in cols_outliers : 
        Q75 = dataset[col].quantile(0.75)
        Q25 = dataset[col].quantile(0.25)
        IQR = Q75-Q25
        upper = Q75 + 3.7 * IQR
        lower = Q25 - 3.7 * IQR
        dataset = dataset[((dataset[col] < upper) & (dataset[col] > lower)) | (dataset[col].isnull()) | (dataset[col] == 0)]
    print(len(dataset)) 
    return dataset

    
def standardize(fit_dataset, transform_dataset):
    """
    Standardize numerical data in the transform_dataset based on the scaling parameters
    learned from the fit_dataset.

    Parameters:
    - fit_dataset: DataFrame used to fit the scaler.
    - transform_dataset: DataFrame where the numerical columns are to be standardized.

    Returns:
    - DataFrame: A copy of the transform_dataset with standardized numerical columns.
    """
    cols_std = [
        #"construction_year",
        "total_area_sqm", 
        "surface_land_sqm",
        "terrace_sqm",
        "primary_energy_consumption_sqm",
        "cadastral_income",    
    ]
    
    scaler = StandardScaler()
    scaler.fit(fit_dataset[cols_std])
    
    scaled_data = scaler.transform(transform_dataset[cols_std])
    #scaled_df = pd.DataFrame(scaled_data, columns=cols_std, index=transform_dataset.index)
    transform_dataset[cols_std] = scaled_data
    return transform_dataset


def train():
    """Trains a linear regression model on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")
    

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
        "garden_sqm",
        "zip_code"
    ]

    fl_features = [
        "fl_terrace",
        "fl_open_fire",
        "fl_swimming_pool",
        "fl_garden",
        "fl_double_glazing",
        #"fl_floodzone", 
        #"fl_furnished"
    ]

    cat_features = [
        #"property_type"
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

    # Impute  numerical features missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])


    #Standardize the numerical features
    #X_train[num_features] = standardize(X_train, X_train)
    #X_test[num_features] = standardize(X_train, X_test)
    
    
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

    #remove outliers 
    #X_train = ouliers(X_train)
    #X_test = outliers(X_test)


    # Define the hyperparameters grid for Gradient Boosting Regressor cross validation
    param_grid = {
        'n_estimators': [150,175, 200],
        'max_depth': [7,8,9],
    }

    # Instantiate the Gradient Boosting Regressor
    model = GradientBoostingRegressor(random_state=505)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model and its parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Parameters:", best_params)
    print("Best R² score (on training set):", best_score)

    # Make predictions on the train set
    y_pred_train = best_model.predict(X_train)
    r2_train = r2_score(y_train, y_pred_train)
    print(f"Train R² score: {r2_train}")

    # Make predictions on the test set
    y_pred_test = best_model.predict(X_test)
    r2_test = r2_score(y_test, y_pred_test)
    print(f"Test R² score: {r2_test}")

    # Save the best model
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
        },
        "imputer": imputer,
        "enc": enc,
        "model": best_model,
    }
    joblib.dump(artifacts, "models/Gradient_boost_artifacts.joblib")


if __name__ == "__main__":
    train()
