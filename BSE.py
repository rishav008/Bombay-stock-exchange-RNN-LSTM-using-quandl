
import quandl
import numpy as np
import matplotlib.pyplot as plt


dataset_train = quandl.get("XNSE/BAJAJ_AUTO", authtoken="k12nM2NDyF9jpZhYsJUy", returns="np")
training_set = dataset_train.iloc[0:2461,1:2].values


from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


X_train = []
y_train = []
for i in range(100, 2461):
    X_train.append(training_set_scaled[i-100:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)


X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout


regressor = Sequential()


regressor.add(LSTM(units = 100, return_sequences = True, input_shape = (None, 1)))
regressor.add(Dropout(0.15))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.15))

regressor.add(LSTM(units = 100, return_sequences = True))
regressor.add(Dropout(0.15))

regressor.add(LSTM(units = 100))


regressor.add(Dense(units = 1))


regressor.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')


regressor.fit(X_train, y_train, epochs = 100, batch_size = 30)


dataset_test = quandl.get("XNSE/BAJAJ_AUTO", authtoken="k12nM2NDyF9jpZhYsJUy", returns="np")
test_set = dataset_test.iloc[2461:,1:2].values
real_stock_price = np.concatenate((training_set[0:2461], test_set), axis = 0)


scaled_real_stock_price = sc.fit_transform(real_stock_price)
inputs = []
for i in range(2461, 2499):
    inputs.append(scaled_real_stock_price[i-100:i, 0])
inputs = np.array(inputs)
inputs = np.reshape(inputs, (inputs.shape[0], inputs.shape[1], 1))
predicted_stock_price = regressor.predict(inputs)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)


plt.plot(real_stock_price[2461:], color = 'red', label = 'Real BAJAJ AUTO stock price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted BAJAJ AUTO Stock Price')
plt.title('BAJAJ AUTO Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('BAJAJ AUTO Stock Price')
plt.legend()
plt.show()