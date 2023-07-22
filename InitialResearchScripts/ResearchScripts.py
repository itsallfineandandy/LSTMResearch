from SimulCapitalQuantUtility.analyticsfunction import get_rsi, get_macd, frac_diff, get_stochastic_oscillator
from SimulCapitalQuantUtility.datafunctions import getMT4data
import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score
from sklearn import preprocessing
import math
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# Parameters
TradingAsset = ['US500']
TimeFrame = 'H1'
end_time = datetime.datetime(2023, 7, 21, 00, 00)
start_time = end_time - datetime.timedelta(days=600)  # last 30 days

# Pull Data
data = getMT4data(TradingAsset, TimeFrame, start_time, end_time)
data['datetime'] = data.index
data = data.sort_index()

# Group and calc data
Asset_group = data.groupby("Asset")
# scaler = MinMaxScaler()
scaler = StandardScaler()

# training split
training_set = data.iloc[:8000, 1:2].values
test_set = data.iloc[8000:, 1:2].values

# Feature Scaling
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
# Creating a data structure with 60 time-steps and 1 output
X_train = []
y_train = []
for i in range(60, 8000):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


model = Sequential()
#Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
model.add(Dropout(0.2))
# Adding a second LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a third LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
# Adding a fourth LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50))
model.add(Dropout(0.2))
# Adding the output layer
model.add(Dense(units = 1))
# Compiling the RNN
model.compile(optimizer = 'adam', loss = 'mean_squared_error')
# Fitting the RNN to the Training set
model.fit(X_train, y_train, epochs = 100, batch_size = 32)

# Getting the predicted stock price of 2017
inputs = test_set.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, len(test_set)):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print(X_test.shape)
# (459, 60, 1)

# Getting the predicted stock price
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(test_set, color = "red")
plt.plot(predicted_stock_price, color = "blue")
plt.show()