import click
import joblib
import pandas as pd
from sklearn.metrics import r2_score


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
    artifacts = joblib.load("models/Linear_artifacts.joblib")

    # Unpack the artifacts
    num_features = artifacts["features"]["num_features"]
    fl_features = artifacts["features"]["fl_features"]
    cat_features = artifacts["features"]["cat_features"]
    imputer = artifacts["imputer"]
    enc = artifacts["enc"]
    model = artifacts["model"]
    standardize = artifacts["standardize"]

    # Extract the used data
    data = data[num_features + fl_features + cat_features]

    # Apply imputer and encoder on data
    data[num_features] = imputer.transform(data[num_features])
    data_cat = enc.transform(data[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    data = pd.concat(
        [
            data[num_features + fl_features].reset_index(drop=True),
            pd.DataFrame(data_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
    )
    # Apply standardize function on data 
    data = standardize(data, data)
    print(data.head())

    # Make predictions
    predictions = model.predict(data)
    #predictions = predictions[:10]  # just picking 10 to display sample output :-)  
    
    #print score
    y_true = pd.read_csv("data/properties.csv")["price"]
    r2_pred = r2_score(y_true, predictions)
    


    ### -------- DO NOT TOUCH THE FOLLOWING LINES -------- ###
    # Save the predictions to a CSV file (in order of data input!)
    pd.DataFrame({"predictions": predictions}).to_csv(output_dataset, index=False)

    # Print success messages
    click.echo(click.style("Predictions generated successfully!", fg="green"))
    click.echo(f"Saved to {output_dataset}")
    click.echo( f"Nbr. observations: {data.shape[0]} | Nbr. predictions: {predictions.shape[0]}")
    print(f"Value of r2_pred: {r2_pred}")
    click.echo( f"R2 pred : {r2_pred}")
    
    ### -------------------------------------------------- ###



if __name__ == "__main__":
    # how to run on command line:
    # python .\predict.py -i "data\properties.csv" -o "output\predictions.csv"
    predict()

