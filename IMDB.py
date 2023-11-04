# 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative).
# Reviews have been preprocessed, and each review is encoded as a sequence of word indexes
# words are indexed by overall frequency, so that for instance the integer "3" encodes the 3rd most frequent
from pandas import read_csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, SimpleRNN
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt


num_words, max_len, embed_len = 1024, 256, 128
data = read_csv('imdb.csv')
x_train, x_test, y_train, y_test = train_test_split(data['review'].to_numpy(),
                                                    data['sentiment'].replace({'negative': 0, 'positive': 1}).to_numpy(),
                                                    test_size=0.3)
t = Tokenizer(num_words)
t.fit_on_texts(x_train)
encoded_x_train = t.texts_to_sequences(x_train)
encoded_x_test = t.texts_to_sequences(x_test)
x_train = pad_sequences(encoded_x_train, maxlen=max_len)
x_test = pad_sequences(encoded_x_test, maxlen=max_len)
model = Sequential()
model.add(Embedding(num_words, output_dim=embed_len, input_length=max_len))
model.add(SimpleRNN(32))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=10, validation_split=0.1)
print(model.summary())
print(model.evaluate(x_test, y_test))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.epoch, history.history['accuracy'], label='Train accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], label='Validation accuracy')
plt.legend()
plt.show()


# stacked rnn
# bidirectional
# GRU
# LSTM




from pandas import read_csv
from numpy import array
from matplotlib.pyplot import plot, show, legend, xlabel, ylabel
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


df = read_csv('jena_climate_2009_2016.csv')['T (degC)']
# Because the data is recorded every 10 minutes, you get 144 data points per day.
sequence_data = []
for i in range(0, 144*356*5, 144):  # data for 5 years
    sequence_data.append(df[i: i + 144].mean())  # avg per day
del df
window_size = 14
data = []
labels = []
for i in range(len(sequence_data) - window_size):
    data.append(sequence_data[i: i + window_size])
    labels.append(sequence_data[i + window_size])
data, labels = array(data), array(labels)
del sequence_data
split_point = len(data) - 100  # 100 points for test
x_train, y_train, x_test, y_test = data[:split_point], labels[:split_point], \
                                   data[split_point:], labels[split_point:]
scalar = StandardScaler().fit(x_train)
x_train = scalar.transform(x_train)
x_test = scalar.transform(x_test)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)
print(x_train.shape) # (1666, 14, 1)
model = Sequential()
model.add(SimpleRNN(32, input_shape=(window_size, 1)))
model.add(Dense(1))
model.compile(optimizer=Adam(learning_rate=0.01), loss="mse")
history = model.fit(x_train, y_train, epochs=100)
print(model.evaluate(x_test, y_test))
predicts = model.predict(x_test)
print(predicts.shape)
xlabel('Day')
ylabel('Weather forecast')
plot(y_test, label='Real')
plot(predicts[:,0], label='Predict')
legend()
show()


