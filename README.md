# Regression

- Repository: `immo-eliza-ml`
- Type: `Consolidation`
- Duration: `3-4 days`
- Deadline: `23/02/2024 5:00 PM`
- Team: solo



## The Mission

The real estate company  asked to create a machine learning model to predict prices of real estate properties in Belgium.

After the **scraping**, **cleaning** and **analyzing**, the objective of the repo is to preprocess the data and build a performant machine learning model!

## Steps followed

### Data preprocessing

 `data/properties.csv` is the starting dataset. These are some notes:
- There are about 76 000 properties, roughly equally spread across houses and apartments
- Each property has a unique identifier `id`
- The target variable is `price`
- Variables prefixed with `fl_` are dummy variables (1/0)
- Variables suffixed with `_sqm` indicate the measurement is in square meters
- All missing categories for the categorical variables are encoded as `MISSING`


- NaNs have been handled through **imputation** 
- categorical data  have been converted into numeric features (**one-hot encoding**)
- numeric features have been rescaled (**standardization**)

Same steps will be of course follow up on the test dataset >> _reusable pipeline_!


Then dataset has been splitted for training and testing.

### Model training, evaluation and iteration

Several model has been trained and  tested from **linear regression** to **RandomForestRegressor**. See the **MODELS_CARD.md** for the output. 

Iteration of the models have been done by testing the impact of outliers management and features engineering change impact on  performance results 

Gridseach has been used to identify potential best parameters for the model. 


## Quotes

_"Artificial intelligence, deep learning, machine learning — whatever you're doing, if you don't understand it — learn it. Because otherwise you're going to be a dinosaur within 3 years." - Mark Cuban_

![You've got this!](https://media.giphy.com/media/5wWf7GMbT1ZUGTDdTqM/giphy.gif)
ml