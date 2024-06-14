
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, Flatten, Reshape, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from keras_tuner import HyperModel

class AutoencoderHyperModel(HyperModel):
    def __init__(self, input_dim):
        self.input_dim = input_dim

    def build(self, hp):
        input_layer = Input(shape=(self.input_dim,))
        if hp.Boolean("use_convolutional"):
            x = Reshape((self.input_dim, 1))(input_layer)
            x = Conv1D(hp.Int('conv_1_units', 16, 64, step=16), 3, activation='relu', padding='same')(x)
            x = MaxPooling1D(2, padding='same')(x)
            x = Dropout(hp.Float('dropout_conv_1', min_value=0.0, max_value=0.5, step=0.05))(x)
            x = Conv1D(hp.Int('conv_2_units', 16, 64, step=16), 3, activation='relu', padding='same')(x)
            encoded = MaxPooling1D(2, padding='same')(x)
            x = Dropout(hp.Float('dropout_conv_2', min_value=0.0, max_value=0.5, step=0.05))(x)
            x = Conv1D(hp.Int('conv_3_units', 16, 64, step=16), 3, activation='relu', padding='same')(encoded)
            x = UpSampling1D(2)(x)
            x = Dropout(hp.Float('dropout_conv_3', min_value=0.0, max_value=0.5, step=0.05))(x)
            x = Conv1D(hp.Int('conv_4_units', 16, 64, step=16), 3, activation='relu', padding='same')(x)
            x = UpSampling1D(2)(x)
            x = Dropout(hp.Float('dropout_conv_4', min_value=0.0, max_value=0.5, step=0.05))(x)
            x = Flatten()(x)
            decoded = Dense(self.input_dim, activation='linear')(x)
        else:
            encoded = Dense(hp.Int('dense_units', 64, 256, step=64), activation='relu',
                            kernel_regularizer=l2(hp.Float('l2_dense', min_value=1e-5, max_value=1e-2, sampling='log')))(input_layer)
            encoded = Dropout(hp.Float('dropout_dense', min_value=0.0, max_value=0.5, step=0.05))(encoded)
            decoded = Dense(self.input_dim, activation='linear')(encoded)

        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer=Adam(hp.Float('learning_rate', 1e-4, 1e-2, sampling='log')),
                            loss='mean_squared_error')
        return autoencoder
