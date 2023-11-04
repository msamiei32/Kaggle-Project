from pandas import read_csv
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras_preprocessing.sequence import pad_sequences
from keras.layers import Dense, Embedding, SimpleRNN, LSTM
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
from keras.optimizers import Adam

data = read_csv('bbc-text.csv')
# print(data['category'].value_counts())
# sport            511
# business         510
# politics         417
# tech             401
# entertainment    386

# plt.hist(data['text'].map(lambda x: len(x.split())), bins=2000)
# plt.show() # test padding = 600, 800, 1000

# t = Tokenizer()
# t.fit_on_texts(data['text'])
# encoded_x_train = t.texts_to_sequences(data['text'])
# print(len(t.word_index)) # 29726
# temp = list(sorted(t.word_counts.values(), reverse=True))
# print(temp[1000])#115
# print(temp[2500])#41
# print(temp[5000])#16
# print(temp[7500])#9
# print(temp[10000])#5
# print(temp[15000])#2
# plt.hist(t.word_counts.values(), bins=2000)
# plt.show()

num_words, max_len, embed_len = 1000, 600, 100
labels = data['category'].replace({'sport': 0, 'business': 1, 'politics': 2, 'tech': 3, 'entertainment': 4})
x_train, x_test, y_train, y_test = train_test_split(data['text'].to_numpy(), labels, test_size=0.3)

t = Tokenizer(num_words)
t.fit_on_texts(x_train)
encoded_x_train = t.texts_to_sequences(x_train)
encoded_x_test= t.texts_to_sequences(x_test)
x_train = pad_sequences(encoded_x_train, maxlen=max_len)
x_test = pad_sequences(encoded_x_test, maxlen=max_len)
model = Sequential()
model.add(Embedding(num_words, output_dim=embed_len, input_length=max_len))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer=Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=50, validation_split=0.1)
print(model.evaluate(x_test, y_test))
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.plot(history.epoch, history.history['loss'], label='Train loss')
plt.plot(history.epoch, history.history['val_loss'], label='Validation loss')
plt.legend()
plt.show()
