import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_csv(r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights_weather_dataset.csv")

# Cleaning the dataset
object_list = [["ECTRL ID", "ADEP", "ADES", "AC Registration", "ACTUAL OFF BLOCK TIME" , "ACTUAL ARRIVAL TIME"],
               ["FILED OFF BLOCK TIME", "FILED ARRIVAL TIME"],
               ["AC Type", "AC Operator", "ICAO Flight Type", "STATFOR Market Segment"]]

# Dropping the unneeded columns
dataset = dataset.drop(columns=object_list[0], axis=1)

# Converting the datetime object columns into different int columns
for col in object_list[1]:
    # Converting the datetime object columns into datetime format
    dataset[col] = pd.to_datetime(dataset[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
    dataset[f"{col}_month"] = dataset[col].dt.month
    dataset[f"{col}_dayofweek"] = dataset[col].dt.dayofweek
    dataset[f"{col}_hour"] = dataset[col].dt.hour
    dataset[f"{col}_minute"] = dataset[col].dt.minute

    # Creating new columns for month, day of week, hour, and minute
    dataset[col + "_month"] = dataset[col].dt.month
    dataset[col + "_dayofweek"] = dataset[col].dt.dayofweek
    dataset[col + "_hour"] = dataset[col].dt.hour
    dataset[col + "_minute"] = dataset[col].dt.minute

# Dropping the original datetime object columns
dataset = dataset.drop(columns=object_list[1], axis=1)

# Converting the string object columns into int format
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
for col in object_list[2]:
    dataset[col] = encoder.fit_transform(dataset[[col]])

# Splitting the dataset into train 60%, validation 20%, and test 20%
trainvalidation, test = train_test_split(dataset, test_size=0.2)
train, validation = train_test_split(trainvalidation, test_size=0.25)

# Removing the target columns from the features and creating separate target datasets
target = ["DEP_DELAY_MIN", "ARR_DELAY_MIN"]
target_index = [dataset.columns.get_loc(col) for col in target]
train_features = train.drop(train.columns[target_index], axis=1)
train_target = train[target]
validation_features = validation.drop(validation.columns[target_index], axis=1)
validation_target = validation[target]
test_features = test.drop(test.columns[target_index], axis=1)
test_target = test[target]

# Evaluation module and printing results
def evaluate(regressor, features, targets):
    predictions = regressor.predict(features)
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    print(f'Mean Absolute Error: {mae:.2f}')
    print(f'Mean Squared Error: {mse:.2f}')
    print(f'R^2 Score: {r2:.2f}')

# Validation module and printing results
def validate(regressor):
    print('Training results:')
    evaluate(regressor, train_features, train_target)
    print("-"*30)
    print('Validation results:')
    evaluate(regressor, validation_features, validation_target)

#
dt = DecisionTreeRegressor()
dt.fit(train_features, train_target)

validate(dt)
print("-"*30)
print("Testing results:")
evaluate(dt, test_features, test_target)

# Showing the top 10 most important features for the model
importances = pd.Series(dt.feature_importances_, index=train_features.columns)
print(importances.sort_values(ascending=False).head(10))
