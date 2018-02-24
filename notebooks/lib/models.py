# LSTM model
import keras.backend as K
from keras.layers import *
from keras.models import *
from keras.optimizers import *


def compute_ctc_cost(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


class WordModel:
    def __init__(self, image_height=100, image_width=200, 
        max_word_length=21, vocabulary_size=81,
        conv_filters=15,
        kernel_width=3,
        pool_size=2,
        time_dense_size=50,
        rnn_size=500):
        
        kernel_size = (kernel_width, kernel_width)

        input_shape = (image_width, image_height, 1)

        the_input = Input(shape=input_shape, dtype='float32', name='the_input')
        x = Conv2D(conv_filters, kernel_size, padding='same', activation='relu', name='conv1')(the_input)
        x = MaxPooling2D(pool_size=(pool_size, pool_size), name='pool2')(x)
        x = Conv2D(conv_filters, kernel_size, padding='same', activation='relu', name='conv3')(x)
        x = MaxPooling2D(pool_size=(pool_size, pool_size), name='pool4')(x)

        conv_to_rnn_dims = (image_width // (pool_size ** 2), (image_height // (pool_size ** 2)) * conv_filters)
        x = Reshape(target_shape=conv_to_rnn_dims, name='reshape')(x)
        
        x = Dense(time_dense_size, activation='relu', name='dense5')(x)
        x = Bidirectional(GRU(rnn_size, return_sequences=True, name='bigru6'))(x)
        x = Bidirectional(GRU(rnn_size, return_sequences=True, name='bigru7'))(x)
        x = Dense((vocabulary_size), kernel_initializer='he_normal', name='dense8')(x)
        y_pred = Activation('softmax', name='softmax')(x)

        labels = Input(name='the_labels', shape=[max_word_length], dtype='float32')
        input_length = Input(name='input_length', shape=[1], dtype='int64')
        label_length = Input(name='label_length', shape=[1], dtype='int64')

        
        loss_out = Lambda(compute_ctc_cost, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

        sgd = SGD(lr=0.02, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
        
        model = Model(inputs=[the_input, labels, input_length, label_length], outputs=loss_out)
        
        # the loss calc occurs elsewhere, so use a dummy lambda func for the loss
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
        self._model = model

    
    def fit_generator(self, *args, **kwargs):
        self._model.fit_generator(*args, **kwargs)
    
    def get_keras_model(self):
        return self._model
    
    def save(self, file_path):
        return self._model.save(file_path)
    
 