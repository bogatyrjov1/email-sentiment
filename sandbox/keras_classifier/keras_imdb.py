from __future__ import print_function

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.datasets import imdb
import numpy as np
import os.path

# Load pretrained model, if exists
if(os.path.isfile("keras_imdb.h5")):
    model = load_model('keras_imdb.h5')
else:
    # set parameters:
    max_features = 5000
    maxlen = 400
    batch_size = 32
    embedding_dims = 50
    filters = 250
    kernel_size = 3
    hidden_dims = 250
    epochs = 1

    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    print('Build model...')
    model = Sequential()

    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen))
    model.add(Dropout(0.2))

    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    # we use max pooling:
    model.add(GlobalMaxPooling1D())

    # We add a vanilla hidden layer:
    model.add(Dense(hidden_dims))
    model.add(Dropout(0.2))
    model.add(Activation('relu'))

    # We project onto a single unit output layer, and squash it with a sigmoid:
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test))

    model.save('keras_imdb.h5')

# Predict sentiment of a sentence
tk = Tokenizer(
    num_words=2000,
    filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
    lower=True,
    split=" ")

text = [
    "Hello there. A wonderful serenity has taken possession of my entire soul, \
    like these sweet mornings of spring which I enjoy with my whole heart. \
    I am alone, and feel the charm of existence in this spot, which was created for the bliss of souls like mine. \
    I am so happy, my dear friend, so absorbed in the exquisite sense of mere tranquil existence, \
    that I neglect my talents. I should be incapable of drawing a single stroke at the present moment; \
    and yet I feel that I never was a greater artist than now. When, while the lovely valley teems with vapour \
    around me, and the meridian sun strikes the upper surface of the impenetrable foliage of my trees, \
    and but a few stray gleams steal into the inner sanctuary, I throw myself down among the tall grass by the \
    trickling stream; and, as I lie close to the earth, a thousand unknown plants are noticed by me: \
    when I hear the buzz of the little world among the stalks, and grow familiar with the countless indescribable \
    forms of the insects and flies, then I feel the presence of the Almighty, who formed us in his own image, \
    and the breath of that universal love which bears and sustains us, as it floats around us in an eternity of bliss; \
    and then, my friend, when darkness overspreads my eyes, and heaven and earth seem to dwell in my soul and \
    absorb its power, like the form of a beloved mistress, then I often think with longing, Oh, would I could \
    describe these conceptions, could impress upon paper all that is living so full and warm within me, that \
    it might be the mirror of my soul, as my soul is the mirror of the infinite God! O my friend -- but it is \
    too much for my strength -- I sink under the weight of the splendour of these visions!A wonderful serenity \
    has taken possession of my entire soul, like these sweet mornings of spring which I enjoy with my whole heart. \
    I am alone, and feel the charm of existence in this spot, which was created for the bliss of souls like mine. \
    I am so happy, my dear friend, so absorbed in the exquisite sense of mere tranquil existence, \
    that I neglect my talents. I should be incapable of drawing a single stroke at the present moment; and"
]
print(text)

tk.fit_on_texts(text)
text_to_sequences = tk.texts_to_sequences(text)
text_to_sequences_np_array = np.array(text_to_sequences)
prediction = model.predict(text_to_sequences_np_array)
print("Sentiment: ", prediction)
