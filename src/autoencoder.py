from keras.layers import Dropout, BatchNormalization
from keras.layers import Input, Dense, GaussianNoise
from keras.models import Model
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint


class AutoEncoder:
    def __init__(self, data_dim, hidden_dim, batch_normalize=True, epoch=100, batch_size=256, loss='binary_crossentropy',
                 optimizer='ADAM', save_name='temp', verbose=1, save_model = False):
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.batch_normalize = batch_normalize
        self.epoch = epoch
        self.batch_size = batch_size
        self.loss = loss
        self.optimizer = optimizer
        self.save_name = save_name
        self.save_model = save_model
        self.verbose = verbose
        self.model = None
        self.embedding_model = None

    def auto_encoder_model(self):
        print(self.hidden_dim)

        init = 'glorot_uniform'
        inputs = Input(shape=(self.data_dim,), name='z')
        # ran = tf.random_normal(shape=(self.data_dim,), mean=0, stddev=0.1)
        x = inputs
#        x = GaussianNoise(stddev=0.1)(x)
        for j, i in enumerate(self.hidden_dim[:-1]):
            x = Dense(i, kernel_initializer=init, name='encoder_%d' % j)(x)
            if self.batch_normalize:
                x = BatchNormalization()(x)
            x = keras.layers.ReLU()(x)
        embedding_layer = Dense(self.hidden_dim[-1], kernel_initializer=init, name='embedding_layer')(x)
        x = embedding_layer
        for j, i in enumerate(self.hidden_dim[1::-1]):
            x = Dense(i, kernel_initializer=init, name='decode_%d' % j)(x)
            if self.batch_normalize:
                x = BatchNormalization()(x)
            x = keras.layers.ReLU()(x)  # ReLU,ELU
        x = Dense(self.data_dim, kernel_initializer=init, activation='sigmoid', name='decoder_0')(x)
        decode = x
        model = Model(inputs=inputs, outputs=decode)
        embedding_model = Model(inputs=inputs, outputs=embedding_layer)
        # plot_model(model, to_file='autoencoder.png', show_shapes=True)
        self.model = model
        self.embedding_model = embedding_model

    def fit(self, x_train, patience=10):
        self.model.compile(loss=self.loss, optimizer=self.optimizer)
        early_stopping = EarlyStopping(monitor='loss', patience=patience)
        callback_name = [early_stopping]
        if self.save_model:
            path_name = self.save_name + " -{epoch:02d}-{loss:.2f}.hdf5"
            modelcheckpoint = ModelCheckpoint(path_name, monitor='loss', save_best_only=True)
            callback_name.append(modelcheckpoint)
        self.model.fit(x_train, x_train, epochs=self.epoch, batch_size=self.batch_size,
                       callbacks=callback_name, verbose=self.verbose)
        encoder_output = self.embedding_model.predict(x_train)
        # if to_save_model:
        #     aeTSNE_utils.save_keras_model(encoder, save_name)
        return encoder_output

    def predict(self, x):
        return self.embedding_model.predict(x)


if __name__ == '__main__':
    pass
#    x_train, y_train = aeTSNE_utils.load_data('mnist')
#    ae = AutoEncoder(784, [500, 500, 2000, 10], epoch=1)
#    ae.auto_encoder_model()
#    embed = ae.fit(x_train)
