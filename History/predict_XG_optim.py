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

    # Convert data to DMatrix format -  not necessary with cross validation 
    #ddata = xgb.DMatrix(data_processed)


    # Make predictions
    predictions = model.predict(data_processed)
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

