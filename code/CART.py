import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dataset = pd.read_csv(r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights_weather_dataset.csv")

# Cleaning the dataset
object_list = [["FILED OFF BLOCK TIME", "FILED ARRIVAL TIME", "ACTUAL OFF BLOCK TIME", "ACTUAL ARRIVAL TIME"],
               ["ADEP", "ADES", "AC Type", "AC Operator", "AC Registration", "ICAO Flight Type", "STATFOR Market Segment"]]

# Converting the datetime columns in to integer format
dataset[object_list[0]] = pd.to_datetime(dataset[object_list[0]], format='%Y-%m-%d %H:%M:%S', errors='coerce')
dataset[object_list[0]] = dataset[object_list[0]].astype(np.int64) // 10**9

print(dataset.info())

# Splitting the dataset into train 60%, validation 20%, and test 20%
trainvalidation, test = train_test_split(dataset, test_size=0.2)
train, validation = train_test_split(trainvalidation, test_size=0.25)

# Removing the target columns from the features and creating separate target datasets
target = ["DEP_DELAY_MIN", "ARR_DELAY_MIN"]
target_index = [18, 19]
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
    print('Validation results:')
    evaluate(regressor, validation_features, validation_target)

#
dt = DecisionTreeRegressor()
dt.fit(train_features, train_target)

print("Training results:")
validate(dt)

print("-"*30)
print("Testing results:")
evaluate(dt, test_features, test_target)