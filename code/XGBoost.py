import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder

dataset = pd.read_csv(r"C:\Users\Gruppe3\Desktop\Machine-Learning-project1\dataset\flights_weather_dataset.csv")

# Cleaning the dataset
object_list = [["ECTRL ID", "ADEP", "ADES", "AC Registration", "ACTUAL OFF BLOCK TIME" , "ACTUAL ARRIVAL TIME", "Actual Distance Flown (nm)"],
               ["FILED OFF BLOCK TIME", "FILED ARRIVAL TIME"],
               ["AC Type", "AC Operator", "ICAO Flight Type", "STATFOR Market Segment"]]

# Dropping the unneeded columns
dataset = dataset.drop(columns=object_list[0], axis=1)

# Converting the datetime object columns into different int columns
for col in object_list[1]:
    # Converting the datetime object columns into datetime format
    dataset[col] = pd.to_datetime(dataset[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
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

# removing the target columns from the features and creating separate target  variable
target = ["DEP_DELAY_MIN", "ARR_DELAY_MIN"]
x = dataset.drop(columns=target, axis=1)
y = dataset[target]

# Splitting the dataset into train 80% and test 20%
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Setting up K for the folds in the cross-validation
cross_validation = KFold(n_splits=9, shuffle=True)

# Setting up the XGBoost model with specified hyperparameters
model = XGBRegressor(learning_rate=0.2 , max_depth=11, colsample_bytree=0.9, n_estimators=500)

# Performing cross-validation using XGBoost
validation_results = cross_validate(model, x_train, y_train, cv=cross_validation,
                                        scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
                                        n_jobs=-1)

# Printing the results of the cross-validation
validation_result = pd.DataFrame(validation_results).mean()
print("Cross-validation results:")
print(f"Mean Absolute Error: {-validation_result['test_neg_mean_absolute_error']:.2f}")
print(f"Mean Squared Error: {-validation_result['test_neg_mean_squared_error']:.2f}")
print(f"R^2 Score: {validation_result['test_r2']:.2f}")
print("-"*30)


# Fitting the model on the training data
model.fit(x_train, y_train)

# Evaluating the model on the test data
y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Printing the evaluation results
print("Test results:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")
