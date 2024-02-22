# Model card

## Project context

The objective is to determine the best possible machine learning model to predict property price on the belgium market. The initial data set is coming from  an initial immoweb scraping,  analysis and cleaning (see (repository)[https://github.com/slvg01/immo-eliza-scraping-Qbicle.git]

## Data
The Input data set has 75K properties over 28 features, beyond the target variable which is the price 
The Data have been preprocessed in order to : 
- remove the outliers. Seeing the nb of outliers a conservative approach has been used to avoid too much data exclusion  wit a +-3.5 IQR usage which induced a loss of 10% of the data.
- standardize the numerical data using standardscaler

## Model details
|Model                     | Parameter                                                             | Results (on test_set) |
| ------------------------ | --------------------------------------------------------------------- | ----------------------|
| Linear Regression        | without Feature optimization / without outliers eclusion              | 0.25                  |
| Linear Regression        | with Features List optimzation / without outliers  exclusion          | 0.35                  |
| Linear Regression        | with outliers  exclusion                                              | 0.41                  |
| Linear Regression        | and with standardization                                              | 0.41                  |
| XGBoost                  | without Feature optimization / without outliers eclusion              | 0.66                  |
| XGBoost                  | with Features List optimzation / without outliers  exclusion          | 0.73                  |
| XGBoost                  | with outliers and standardization                                     | 0.76                  |
| XGBoost                  | with outliers and standardization                                     | 0.41                  |
| XGBoost                  | with outliers and standardization                                     | 0.41                  |




## Performance

Performance metrics for the various models tested, visualizations, ...

## Limitations

What are the limitations of your model?

## Usage

What are the dependencies, what scripts are there to train the model, how to generate predictions, ...

## Maintainers

Who to contact in case of questions or issues?
