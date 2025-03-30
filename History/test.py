import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv("data/properties.csv")


num_cols =[
"total_area_sqm", 
"surface_land_sqm",
"nbr_frontages",
"nbr_bedrooms",
"terrace_sqm",
"garden_sqm",
"primary_energy_consumption_sqm",
"cadastral_income",
]
### number of numerical features
N_num = len(num_cols)

### height of figure depends on number of numerical features
plt.figure(figsize=(15, 5*N_num)) 

### for each numerical feature plot a boxplot
for i in np.arange(N_num):
    col = num_cols[i]
    
    plt.subplot(N_num, 1, i+1)
    plt.title(f'Boxplot of {col}')
    
    ### boxplot uses Quartiles, median from data 
    sns.boxplot(data=data, x=col)
                    
### prevent overlap, make bunch of graph look nicer when put together
plt.tight_layout()

plt.show()


"""




"garden_sqm",
"region",
"province",
fl_floodzone
data["fl_epc"] = data["epc"].apply(
        lambda x: 1 if x in ["A", "A+", "A++", "B"] 
        else ( 2 if x =='MISSING' else 0)
    )

    data["equipped_kitchen_synth"] = data["equipped_kitchen"].apply(
        lambda x: "NOT_EQUIPPED"
        if x in ["NOT_INSTALLED", "USA_UNINSTALLED"]
        else "EQUIPPED"
    )
    #print(data.head())


    Q75 = data["garden_sqm"].quantile(0.75)
    Q25 = data["garden_sqm"].quantile(0.25)
    IQR = Q75-Q25
    upper = Q75 + 3.5 * IQR
    lower = Q25 - 3.5 * IQR
    data = data[((data["garden_sqm"] < upper) & (data["garden_sqm"] > lower)) | (data["garden_sqm"].isnull()) | (data["garden_sqm"] == 0)]
    print(len(data))
    

    
"""    # Standardize numerical data 
    num_cols2 = [
    "latitude",
    "longitude",
    "constuction year"
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
    print(data.head(10))



  data = data[(data["garden_sqm"] < 20000)]
    print(len(data))
"""


   """ --------------------------

    # Set parameters for XGBoost
    params = {
    'max_depth': 5,
    'eta': 0.1,
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse'
    }
    
    # Train the model
    num_rounds = 2500
    model = xgb.train(params, dtrain, num_rounds)

    # Make predictions on the train set
    y_pred_train = model.predict(dtrain)

    # Make predictions on the test set
    y_pred_test = model.predict(dtest)

    # Evaluate the model using R2 score
    r2_train = r2_score(y_train, y_pred_train)
    print(f"Train R² score: {r2_train}")
    
    r2_test = r2_score(y_test, y_pred_test)
    print(f"Test R² score: {r2_test}")
"""

_________________________

import click
import joblib
import pandas as pd
import xgboost as xgb


@click.command()
@click.option("-i", "--input-dataset", help="path to input .csv dataset", required=True)
@click.option(
    "-o",
    "--output-dataset",
    default="output/predictions.csv",
    help="full path where to store predictions",
    required=True,
)
def predict(input_dataset, output_dataset):
    """Predicts house prices from 'input_dataset', stores it to 'output_dataset'."""
    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Load the data
    data = pd.read_csv(input_dataset)
    ### -------------------------------------------------- ###

    # Load the model artifacts using joblib
    artifacts = joblib.load("models/XG_boost_artifacts.joblib")

   

    # Unpack the artifacts
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    model = artifacts["model"]

    # Extract the used data
    data_extr = data[num_features + fl_features + cat_features]

   
    # Apply imputer and encoder on data
    data_extr[num_features] = imputer.transform(data_extr[num_features])
    data_cat = enc.transform(data_extr[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    data_processed = pd.concat(
        [
            data_extr[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )

    # Convert data to DMatrix format
    ddata = xgb.DMatrix(data_processed)


    # Make predictions
    predictions = model.predict(ddata)
    #predictions = predictions[:10]  # just picking 10 to display sample output :-)  

    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file (in order of data input!)
    pd.DataFrame({"predictions": predictions}).to_csv(output_dataset, index=False)

    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to {output_dataset}")
    click.echo(
        f"Nbr. observations: {data_processed.shape[0]} | Nbr. predictions: {predictions.shape[0]}"
    )
    ### -------------------------------------------------- ###


if __name__ == "__main__":
    # how to run on command line:
    # python .\predict.py -i "data\properties.csv" -o "output\predictions.csv"
    predict()
__________________________
scaled_data_train = scaler.transform(transform_dataset_train)[cols_std]
    scaled_data_test = scaler.transform(transform_dataset_test)[cols_std]
    #scaled_df = pd.DataFrame(scaled_data, columns=cols_std, index=transform_dataset.index)
    transform_dataset_train[cols_std] = scaled_data_train
    transform_dataset_test[cols_std] = scaled_data_test
    return transform_dataset_train, transform_dataset_test, 