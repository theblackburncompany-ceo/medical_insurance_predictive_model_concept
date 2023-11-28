import pandas as pd

df = pd.read_csv('https://raw.githubusercontent.com/theblackburncompany-ceo/test_datasets/main/neuralnetwork1.csv')
df

y = df['logS']
y

x = df.drop('logS', axis=1)
x


# Data Splitting

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)
x_train


# Model Building
# Linear Regression
# Training the Model
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

# Applying Model for Predictions
y_lr_train_pred = lr.predict(x_train)
y_lr_test_pred = lr.predict(x_test)

y_lr_train_pred

y_lr_test_pred

# Evaluate Model Performance
from sklearn.metrics import mean_squared_error, r2_score

lr_train_mse = mean_squared_error(y_train, y_lr_train_pred)

lr_train_r2 = r2_score(y_train, y_lr_train_pred) 

lr_test_mse = mean_squared_error(y_test, y_lr_test_pred)

lr_test_r2 = r2_score(y_test, y_lr_test_pred)
 

print('LR MSE (Train): ', lr_train_mse)

print('LR MSE (Test): ', lr_test_mse)

print('LR R2 (Train): ', lr_train_r2)

print('LR R2 (Test): ', lr_test_r2) 

lr_results = pd.DataFrame(['Linear regression', lr_train_mse, lr_train_r2, lr_test_mse, lr_test_r2]).transpose()

lr_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

lr_results


# Random Forest
# Training the Model
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(max_depth=2, random_state=100)

rf.fit(x_train, y_train)

# Applying the Model to Make a Prediction
y_rf_train_pred = rf.predict(x_train)

y_rf_test_pred = rf.predict(x_test)

# Evaluate Model Performance
from sklearn.metrics import mean_squared_error, r2_score

rf_train_mse = mean_squared_error(y_train, y_rf_train_pred)

rf_train_r2 = r2_score(y_train, y_rf_train_pred)

rf_test_mse = mean_squared_error(y_test, y_rf_test_pred)

rf_test_r2 = r2_score(y_test, y_rf_test_pred)

rf_results = pd.DataFrame(['Random forest', rf_train_mse, rf_train_r2, rf_test_mse, rf_test_r2]).transpose()

rf_results.columns = ['Method', 'Training MSE', 'Training R2', 'Test MSE', 'Test R2']

rf_results


# Model Comparison

df_models = pd.concat([lr_results, rf_results], axis=0).reset_index(drop=True)

df_models


# Data Visulatization
import matplotlib.pyplot as plt

import numpy as np

plt.figure(figsize=(5,5))

plt.scatter(x=y_train, y=y_rf_train_pred, c='#7CAE00', alpha=1)

z = np.polyfit(y_train, y_rf_train_pred, 1)

p = np.poly1d(z)

plt.plot(y_train, p(y_train), '#F8766D')

plt.ylabel('Predict Logs')

plt.xlabel('Experimental Logs')
