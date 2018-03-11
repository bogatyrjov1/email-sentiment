# Tutorial: https://machinelearningmastery.com/predict-sentiment-movie-reviews-using-deep-learning/

import numpy
# Import IMDB reviews labeled dataset
# Downloads it from https://s3.amazonaws.com/text-datasets/imdb.pkl (33M) on the first run
# Or https://s3.amazonaws.com/text-datasets/imdb.npz
from keras.datasets import imdb
from keras.layers import Embedding
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot

# load the dataset
(X_train, y_train), (X_test, y_test) = imdb.load_data()
X = numpy.concatenate((X_train, X_test), axis=0)
y = numpy.concatenate((y_train, y_test), axis=0)

# summarize size
print("Training data: ")
print(X.shape)
print(y.shape)

# Summarize number of classes
print("Classes: ")
print(numpy.unique(y))

# Summarize number of words
print("Number of words: ")
print(len(numpy.unique(numpy.hstack(X))))

# Summarize review length
print("Review length: ")
result = [len(x) for x in X]
print("Mean %.2f words (%f)" % (numpy.mean(result), numpy.std(result)))

# Create a word embedding representation of IMDB reviews Keras dataset
imdb.load_data(num_words=5000)
X_train = pad_sequences(X_train, maxlen=500)
X_test = pad_sequences(X_test, maxlen=500)
Embedding(5000, 32, input_length=500)