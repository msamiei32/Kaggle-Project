from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from pandas import read_csv
from keras.optimizers import SGD


data = read_csv('ENB2012_data.csv')
y1 = data.Y1.to_numpy()
y2 = data.Y2.to_numpy()
data.drop(['Y1', 'Y2'], axis=1, inplace=True)
split = int(len(data) * 0.8)
x_train, x_test = data[:split], data[split:]
y1_train, y1_test, y2_train, y2_test = y1[:split], y1[split:], y2[:split], y2[split:],

x_train = (x_train - x_train.mean()) / x_train.std()
x_test = (x_test - x_train.mean()) / x_train.std()

input_layer = Input(shape=x_train.shape[1])
first_dense = Dense(units='128', activation='relu')(input_layer)
second_dense = Dense(units='128', activation='relu')(first_dense)
third_dense = Dense(units='128', activation='relu')(second_dense)
y1_output = Dense(units='1', name='y1_output')(third_dense)
y2_output = Dense(units='1', name='y2_output')(third_dense)

model = Model(inputs=input_layer, outputs=[y1_output, y2_output])

model.compile(optimizer=SGD(lr=0.001), loss={'y1_output': 'mse', 'y2_output': 'mse'})

history = model.fit(x_train, [y1_train, y2_train], epochs=50, batch_size=10, validation_split=0.1)

print(model.evaluate(x=x_test, y=[y1_test, y2_test]))

import matplotlib.pyplot as plt
plt.plot(history.history['loss'], color='blue', label='train')
plt.plot(history.history['val_loss'], color='green', label='val')
plt.show()