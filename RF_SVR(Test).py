import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

# Load each dataset
df_temp = pd.read_csv('temperature.csv')
df_humidity = pd.read_csv('humidity.csv')
df_pressure = pd.read_csv('pressure.csv')
df_wind_speed = pd.read_csv('wind_speed.csv')
df_rainfall = pd.read_csv('rainfall.csv')  # This file is similar to the one attached

# Merge the datasets on year, month, and day
df = pd.merge(df_temp, df_humidity, on=['Year', 'Month', 'Day'])
df = pd.merge(df, df_pressure, on=['Year', 'Month', 'Day'])
df = pd.merge(df, df_wind_speed, on=['Year', 'Month', 'Day'])
df = pd.merge(df, df_rainfall, on=['Year', 'Month', 'Day'])

# Convert columns to numeric data types
df['Temperature'] = pd.to_numeric(df['Temperature'])
df['Pressure'] = pd.to_numeric(df['Pressure'])
df['Wind_Speed'] = pd.to_numeric(df['Wind_Speed'])
df['Rainfall'] = pd.to_numeric(df['Rainfall'])

# Use IterativeImputer to fill missing values
imputer = IterativeImputer(random_state=42)
df[['Temperature', 'Humidity', 'Pressure', 'Wind_Speed', 'Rainfall']] = \
    (imputer.fit_transform(df[['Temperature', 'Humidity', 'Pressure', 'Wind_Speed', 'Rainfall']]))

# Define features and target variable
X = df[['Temperature', 'Humidity', 'Pressure', 'Wind_Speed']]
y = df['Rainfall']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 1: Tune C and epsilon
param_grid_1 = {
    'C': [1, 10, 100, 1000],
    'epsilon': [0.1, 0.01, 0.001]
}

svr = SVR(kernel='rbf', gamma='scale')
grid_search_1 = GridSearchCV(svr, param_grid_1, cv=5, scoring='neg_mean_squared_error')
grid_search_1.fit(X_train, y_train)

best_c = grid_search_1.best_params_['C']
best_epsilon = grid_search_1.best_params_['epsilon']

# Step 2: Tune kernel and gamma
param_grid_2 = {
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': [0.1, 1, 10]
}

svr = SVR(C=best_c, epsilon=best_epsilon)
grid_search_2 = GridSearchCV(svr, param_grid_2, cv=5, scoring='neg_mean_squared_error')
grid_search_2.fit(X_train, y_train)

best_kernel = grid_search_2.best_params_['kernel']
best_gamma = grid_search_2.best_params_['gamma']

print('Best kernel:', best_kernel)
print('Best gamma:', best_gamma)
print('Best epsilon:', best_epsilon)
print('Best c:', best_c)

# Train the final model with the tuned hyperparameters
final_model = SVR(C=best_c, epsilon=best_epsilon, kernel=best_kernel, gamma=best_gamma)
final_model.fit(X_train, y_train)

# Predict on the test set
y_pred = final_model.predict(X_test)

# Calculate RMSE
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Example: Predicting with new data
temp = input('Enter the temperature in Celsius: ')
humidity = input('Enter the humidity in % :')
pressure = input('Enter the atmospheric pressure in hPa: ')
wind_speed = input('Enter the wind speed in km/hr: ')

new_data = pd.DataFrame({
    'Temperature': [temp],
    'Humidity': [humidity],
    'Pressure': [pressure],
    'Wind_Speed': [wind_speed]
})
new_rainfall_prediction = final_model.predict(new_data)
print(f'Predicted Rainfall: {new_rainfall_prediction[0]} mm')
