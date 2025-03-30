import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor  # Import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


def outliers(dataset):
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
    for col in cols_outliers:
        Q75 = dataset[col].quantile(0.75)
        Q25 = dataset[col].quantile(0.25)
        IQR = Q75 - Q25
        upper = Q75 + 3.7 * IQR
        lower = Q25 - 3.7 * IQR
        dataset = dataset[
            ((dataset[col] < upper) & (dataset[col] > lower))
            | (dataset[col].isnull())
            | (dataset[col] == 0)
        ]
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

    # Standardize numerical data
    cols_std = [
        "construction_year",
        "total_area_sqm",
        "surface_land_sqm",
        "terrace_sqm",
        "primary_energy_consumption_sqm",
        "cadastral_income",
    ]

    scaler = StandardScaler()
    scaler.fit(fit_dataset[cols_std])

    scaled_data = scaler.transform(transform_dataset[cols_std])
    scaled_df = pd.DataFrame(scaled_data, columns=cols_std, index=transform_dataset.index)
    transform_dataset[cols_std] = scaled_df
    return transform_dataset


def train():
    """Trains a Random Forest regression model on the full dataset and stores output."""
    # Load the data

    data = pd.read_csv("data/properties.csv")
    data = outliers(data)

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

    # Define the parameter grid for cross-validation
    param_grid = {
        "n_estimators": [100, 200, 300],  # Number of trees in the forest
        "max_depth": [None, 10, 20],  # Maximum depth of the tree
        "min_samples_split": [2, 5, 10],  # Minimum samples required to split a node
        "min_samples_leaf": [1, 2, 4],  # Minimum samples required at each leaf node
        "max_features": ["auto", "sqrt", "log2"],  # Number of features to consider for the best split
        "random_state": [505],
    }

    # Create the Random Forest regressor
    rf_reg = RandomForestRegressor()

    # Perform grid search cross-validation
    grid_search = GridSearchCV(
        estimator=rf_reg,
        param_grid=param_grid,
        cv=5,
        scoring="r2",
        verbose=2,
        n_jobs=-1,
    )
    grid_search.fit(X_train, y_train)

    # Get the best parameters and best estimator
    best_params = grid_search.best_params_
    best_model = grid_search.best_estimator_

    # Make predictions on the test set
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # Evaluate the model using R2 score
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    print(f"Train R² score: {r2_train}")
    print(f"Test R² score: {r2_test}")

    print("Best Parameters:", best_params)

    # Save the model
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
    joblib.dump(artifacts, "models/RandomForest_artifacts.joblib")


if __name__ == "__main__":
    train()
